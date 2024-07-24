
from dataset import MMGraphDataset
from model import create_model
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
import os
import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from typing import Optional
from utils import (
    AverageMeter,
    AUCRecorder,
    accuracy,
    calculate_precision_recall_f1,
    calculate_accuracy_from_metrics
)
import time
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random

from const import (
    GRAPH,
    LABEL,
)
import gc
from shutil import copyfile
from torch.cuda.amp import GradScaler, autocast
from model import FusionMLP, FusionTransformer, FusionMamba
import torch_geometric


class MMTrainer:
    def __init__(
            self,
            config_path: str,
    ):
        config = OmegaConf.load(config_path)

        if config.environment.seed is not None:
            torch.manual_seed(config.environment.seed)
            np.random.seed(config.environment.seed)
            random.seed(config.environment.seed)

        self.device = torch.device('cuda:{}'.format(config.environment.gpu_id) if torch.cuda.is_available() else 'cpu')

        # 动态数据加载
        train_graphs = {'cell_graph_path': config.data.train.cell_graph_path,
                        'patch_graph_lv0_path': config.data.train.patch_graph_lv0_path,
                        'patch_graph_lv1_path': config.data.train.patch_graph_lv1_path,
                        'tissue_graph_path': config.data.train.tissue_graph_path}

        trainset = MMGraphDataset(
            use_cell_graph=config.data.use_cell_graph,
            use_patch_graph_lv0=config.data.use_patch_graph_lv0,
            use_patch_graph_lv1=config.data.use_patch_graph_lv1,
            use_tissue_graph=config.data.use_tissue_graph,
            **train_graphs,
            train_mode=True,
        )

        train_loader = DataLoader(
            dataset=trainset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        # 动态数据加载
        test_graphs = {'cell_graph_path': config.data.test.cell_graph_path,
                       'patch_graph_lv0_path': config.data.test.patch_graph_lv0_path,
                       'patch_graph_lv1_path': config.data.test.patch_graph_lv1_path,
                       'tissue_graph_path': config.data.test.tissue_graph_path}

        testset = MMGraphDataset(
            use_cell_graph=config.data.use_cell_graph,
            use_patch_graph_lv0=config.data.use_patch_graph_lv0,
            use_patch_graph_lv1=config.data.use_patch_graph_lv1,
            use_tissue_graph=config.data.use_tissue_graph,
            **test_graphs,
            train_mode=False,
        )

        test_loader = DataLoader(
            dataset=testset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

        print('Complete loading data.')


        # 构建动态的 base_path
        graph_types = []
        if config.data.use_cell_graph:
            graph_types.append("cell")
        if config.data.use_patch_graph_lv0:
            graph_types.append("patchlv0")
        if config.data.use_patch_graph_lv1:
            graph_types.append("patchlv1")
        if config.data.use_tissue_graph:
            graph_types.append("tissue")
        base_folder = '_'.join(graph_types)
        base_path = f"/root/autodl-tmp/Results/TCGA_STAD/{base_folder}/"
        os.makedirs(base_path, exist_ok=True)

        if not config.output_path.tuning:
            file_name = str(len(os.listdir(base_path)) + 1) + '_' + config.output_path.exp_file
        else:
            file_name = config.output_path.exp_file
        self.output_path = os.path.join(base_path, file_name)
        exist_ok = config.output_path.exist_ok if config.output_path.exist_ok else True
        os.makedirs(self.output_path, exist_ok=exist_ok)
        # 保存当前epoch的模型权重和最佳ACC时候的模型权重
        os.makedirs(os.path.join(self.output_path, 'weights', 'acc'), exist_ok=exist_ok)
        # 保存当前epoch的模型权重和最佳AUC时候的模型权重
        os.makedirs(os.path.join(self.output_path, 'weights', 'auc'), exist_ok=exist_ok)
        # 保存每个epoch结束后对应的ROC曲线，并绘制最佳AUC时的ROC曲线图
        os.makedirs(os.path.join(self.output_path, 'figures'), exist_ok=exist_ok)
        self.logging = open(os.path.join(self.output_path, 'logging.txt'), 'w')
        copyfile(config_path, os.path.join(self.output_path, 'config.yaml'))
        print('Complete creating export files.')


        assert trainset.num_features == testset.num_features
        model = create_model(
            config=config.models,
            num_classes=config.data.num_classes,
            in_features=trainset.num_features)

        self.model = model.to(self.device)
        print('Complete creating model.')



        # 计算并记录模型的可学习参数量
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        params_msg = f"Total trainable parameters: {total_params}\n"
        print(params_msg)
        self.logging.write(params_msg)
        self.logging.flush()



        # 设置优化器 AdamW
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay
        )

        # 设置学习率调度器
        # 指数学习率调度器
        if config.optimization.scheduler.lower() == 'epoential':
            self.scheduler = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=config.optimization.gamma
            )
        # 余弦退火调度器
        elif config.optimization.scheduler.lower() == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=10,
                eta_min=config.optimization.min_learning_rate,
            )
        # 保持学习率不变的调度器
        elif config.optimization.scheduler.lower() == 'constant':
            self.scheduler = lr_scheduler.ConstantLR(
                optimizer=self.optimizer,
            )
        else:
            raise ValueError("Unkown scheduler {}".format(config.optimization.scheduler.lower()))
        self.epochs = config.optimization.epochs
        self.patience = config.optimization.patience

    def train(
            self,
            verbosity: Optional[bool] = True,
    ):
        best_test_acc = 0.0
        best_test_auc = 0.0
        best_epoch_acc = 0
        best_epoch_auc = 0

        # 初始化存储在最佳ACC和AUC时对应的其他指标
        best_metrics_at_best_auc = {}

        time_start = time.time()

        msg = 'Total training epochs : {}\n'.format(self.epochs)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, train_auc = self._train_one_epoch()
            test_loss, test_acc, test_auc, _test_auc_recorder, precision, recall, f1 = self._test_per_epoch(model=self.model)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch_acc = epoch

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch_auc = epoch
                best_metrics_at_best_auc = {
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_path, 'weights', 'auc', 'model_epoch{}.pth'.format(epoch)))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights', 'auc', 'best_model.pth'))

                _test_auc_recorder.draw_roc(
                    path=os.path.join(self.output_path, 'figures', 'epoch_{}_test_roc.png'.format(epoch))
                )

                roc_data_path = os.path.join(self.output_path, 'figures', 'epoch_{}_roc_data.csv'.format(epoch))
                _test_auc_recorder.save_to_file(roc_data_path)

            n_positive = 27
            n_negative = 81
            acc_from_metrics = calculate_accuracy_from_metrics(best_metrics_at_best_auc['Precision'],
                                                               best_metrics_at_best_auc['Recall'], n_positive,
                                                               n_negative)

            msg = (f'Epoch {epoch:03d} ##################'
                   f'\n\tTrain loss: {train_loss:.5f}, Train acc: {train_acc:.3f}%, Train auc: {train_auc* 100:.3f}%;'
                   f'\n\tTest loss: {test_loss:.5f}, Test auc: {test_auc* 100:.3f}%;'
                   f'\n\tBest test auc: {best_test_auc* 100:.3f}%')
            msg += (f'\n\t'f'Acc: {acc_from_metrics* 100:.1f}%, '
                    f''f'Auc: {best_test_auc* 100:.1f}%, '
                    f'Precision: {best_metrics_at_best_auc.get("Precision", 0)* 100:.1f}%, '
                    f'Recall: {best_metrics_at_best_auc.get("Recall", 0)* 100:.1f}%, '
                    f'F1 Score: {best_metrics_at_best_auc.get("F1 Score", 0)* 100:.1f}%')


            if verbosity:
                print(msg)
            self.logging.write(msg)
            self.logging.flush()


            # 提前停止的条件
            if epoch > 0:
                msg = "Stopping early due to no improvement in both acc and auc for {} epochs.\n".format(self.patience)
                print(msg)  # 输出停止信息，并在其后增加一个空行
                self.logging.write(msg)
                self.logging.flush()
                break


            # # 提前停止的条件
            # if (epoch - max(best_epoch_acc, best_epoch_auc)) > self.patience:
            #     msg = "Stopping early due to no improvement in both acc and auc for {} epochs.\n".format(self.patience)
            #     print(msg)  # 输出停止信息，并在其后增加一个空行
            #     self.logging.write(msg)
            #     self.logging.flush()
            #     break


        msg_auc = (f"Best test auc: {best_test_auc:.4f} @ epoch {best_epoch_auc}, "
                   f"Acc: {acc_from_metrics:.3f}, "
                   f"Precision: {best_metrics_at_best_auc.get('Precision', 0):.3f}, "
                   f"Recall: {best_metrics_at_best_auc.get('Recall', 0):.3f}, "
                   f"F1 Score: {best_metrics_at_best_auc.get('F1 Score', 0):.3f}\n")

        if verbosity:
            print(msg_auc)
        self.logging.write(msg_auc)
        self.logging.flush()

        time_end = time.time()
        msg = "run time: {:.1f}s, {:.2f}h\n".format(time_end - time_start, (time_end - time_start) / 3600)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush()

    def _train_one_epoch(self):


        _train_loss_recorder = AverageMeter()
        _train_acc_recorder = AverageMeter()
        _train_auc_recorder = AUCRecorder()
        torch.autograd.set_detect_anomaly(True)
        self.model.train()

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            # 清空当前梯度信息以准备下一批次的训练
            self.optimizer.zero_grad()

            data, label = data

            # 把所有图和标签移至设备
            for key in data:
                data[key] = data[key].to(self.device)
            label = label.to(self.device)

            data[LABEL] = label

            # 模型前向传播
            out = self.model(data)
            assert not torch.isnan(out).any(), "Model outputs NaN during training"

            # 使用模型的输出和正确的标签来计算损失
            loss = F.cross_entropy(out, data[LABEL])
            if torch.isnan(loss).any():
                raise ValueError("Loss is NaN")

            loss.backward()
            # 更新模型参数
            self.optimizer.step()

            acc = accuracy(out, data[LABEL])[0]
            _train_loss_recorder.update(loss.item(), out.size(0))
            _train_acc_recorder.update(acc.item(), out.size(0))
            _train_auc_recorder.update(prediction=out[:, 1], target=data[LABEL])


                # 在一个epoch结束后，根据配置的学习率调度策略更新学习率
        self.scheduler.step()

        train_loss = _train_loss_recorder.avg
        train_acc = _train_acc_recorder.avg
        train_auc = _train_auc_recorder.auc

        return train_loss, train_acc, train_auc

    def _test_per_epoch(self, model):
        _test_loss_recorder = AverageMeter()
        _test_acc_recorder = AverageMeter()
        _test_auc_recorder = AUCRecorder()
        _predictions = []
        _targets = []
        total_inference_time = 0  # 初始化总推理时间
        total_samples = 0  # 初始化总样本数


        with torch.no_grad():
            model.eval()
            for batch_idx, data in enumerate(tqdm(self.test_loader)):
                data, label = data
                for key in data:
                    data[key] = data[key].to(self.device)
                label = label.to(self.device)
                data[LABEL] = label
                start_time = time.time()
                out = model(data)
                end_time = time.time()
                inference_time = end_time - start_time
                total_inference_time += inference_time
                total_samples += label.size(0)


                assert not torch.isnan(out).any(), "Model outputs NaN during testing"

                loss = F.cross_entropy(out, data[LABEL])

                acc = accuracy(out, data[LABEL])[0]
                _test_loss_recorder.update(loss.item(), out.size(0))
                _test_acc_recorder.update(acc.item(), out.size(0))
                _test_auc_recorder.update(prediction=out[:, 1], target=data[LABEL])
                _predictions.extend(out[:, 1].detach().cpu().numpy())
                _targets.extend(label.cpu().numpy())
        test_loss = _test_loss_recorder.avg
        test_acc = _test_acc_recorder.avg
        test_auc = _test_auc_recorder.auc

        precision, recall, f1 = calculate_precision_recall_f1(np.array(_predictions), np.array(_targets))

        average_inference_time_per_sample = total_inference_time / total_samples  # 计算每个样本的平均推理时间

        self.logging.write(f"Total Samples: {total_samples}\n")
        self.logging.write(f"Average inference time per sample: {average_inference_time_per_sample:.4f}s\n")  # 记录到日志
        self.logging.flush()
        return test_loss, test_acc, test_auc, _test_auc_recorder, precision, recall, f1



