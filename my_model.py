
import torch
from torch import nn
from typing import List, Optional, Tuple
from models import MLP
from utils import init_weights
from const import (
    CELL_GRAPH,
    PATCH_GRAPH_LV0,
    PATCH_GRAPH_LV1,
    TISSUE_GRAPH,
    LABEL,
    LOGITS,
    FEATURES,

    CELL_GNN_MODEL,
    PATCH_LV0_GNN_MODEL,
    PATCH_LV1_GNN_MODEL,
    TISSUE_GNN_MODEL,
    FUSION_MLP,
    FUSION_TRANSFORMER,
    FUSION_MAMBA,
)

from models import gnn_model_dict
import timm
import functools
from ft_transformer import FT_Transformer
from typing import Dict
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

class GNN(nn.Module):
    def __init__(
            self,
            prefix: str,
            model_name: str,
            num_classes: int,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):

        super(GNN, self).__init__()   # 调用父类 nn.Module 的构造方法

        self.prefix = prefix
        self.num_classes = num_classes

        # 使用常量定义 data_key
        if self.prefix.lower() == 'cell_gnn':
            self.data_key = CELL_GRAPH
        elif self.prefix.lower() == 'patch_lv0_gnn':
            self.data_key = PATCH_GRAPH_LV0
        elif self.prefix.lower() == 'patch_lv1_gnn':
            self.data_key = PATCH_GRAPH_LV1
        elif self.prefix.lower() == 'tissue_gnn':
            self.data_key = TISSUE_GRAPH
        else:
            raise ValueError(f"Unknown graph_name: {graph_name}")

        self.label_key = f"{LABEL}"

        self.model = gnn_model_dict[model_name](
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            pooling=pooling,
            activation=activation,
        )

        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.out_features = out_features

    def forward(self, batch):
        data = batch[self.data_key]

        features = self.model(data)
        logits = self.head(features)

        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }


class FusionMLP(nn.Module):
    def __init__(
            self,
            prefix: str,
            models: list,
            num_classes: int,
            hidden_features: List[int],
            adapt_in_features: Optional[str] = None,
            activation: Optional[str] = "gelu",
            dropout_prob: Optional[float] = 0.5,
            normalization: Optional[str] = "layer_norm",
    ):

        super().__init__()
        self.prefix = prefix
        self.model = nn.ModuleList(models)

        # TODO: Add out_features to each model
        raw_in_features = [per_model.out_features for per_model in models]

        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )

            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
            )
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)

        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features

        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        self.head = nn.Linear(in_features, num_classes)
        # init weights

        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)

    def forward(
            self,
            batch: dict,
    ):
        multimodal_features = []

        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_features.append(per_adapter(per_output[per_model.prefix][FEATURES]))

        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        logits = self.head(features)
        return logits


class FusionTransformer(nn.Module):

    def __init__(
            self,
            prefix: str,
            models: list,
            hidden_features: int,
            num_classes: int,
            adapt_in_features: Optional[str] = None,
    ):
        super().__init__()

        self.model = nn.ModuleList(models)

        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )

            in_features = base_in_feat
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
            )
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)

        self.fusion_transformer = FT_Transformer(
            d_token=in_features,
            n_blocks=1,
            attention_n_heads=4,
            attention_dropout=0.1,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_d_hidden=192,
            ffn_dropout=0.1,
            ffn_activation='ReGLU',
            ffn_normalization='LayerNorm',
            residual_dropout=0.0,
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=None,
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation='ReLU',
            head_normalization='LayerNorm',
            d_out=hidden_features,
        )

        self.head = FT_Transformer.Head(
            d_in=in_features,
            d_out=num_classes,
            bias=True,
            activation='ReLU',
            normalization='LayerNorm',
        )

        # init weights
        self.adapter.apply(init_weights)
        self.head.apply(init_weights)

        self.prefix = prefix

    def forward(
            self,
            batch: dict,
    ):
        multimodal_features = []
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)

        multimodal_features = torch.cat(multimodal_features, dim=1)
        features = self.fusion_transformer(multimodal_features)
        logits = self.head(features)

        return logits


class FusionMamba(nn.Module):
    def __init__(
            self,
            prefix: str,
            models: list,
            num_classes: int,
            hidden_features: int,
            adapt_in_features: Optional[str] = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.model = nn.ModuleList(models)

        # 计算每个模型输出的特征维度
        raw_in_features = [per_model.out_features for per_model in models]

        # 设置适配器，统一特征维度
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"Unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in raw_in_features]
            )
            base_in_feat = sum(raw_in_features)

        # 初始化Mamba模块
        self.mamba = Mamba(
            d_model=base_in_feat,  # 使用统一的特征维度
            d_state=8,  # 内部状态的维度
            bimamba_type="v1",
            # bimamba_type="v2",
            # if_devide_out=True
        )

        self.head = FT_Transformer.Head(
            d_in=base_in_feat,
            d_out=num_classes,
            bias=True,
            activation='ReLU',
            normalization='LayerNorm',
        )

    def forward(self, batch: dict):
        multimodal_features = []
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)

        multimodal_features = torch.cat(multimodal_features, dim=1)
        # 使用Mamba模块处理特征
        mamba_features = self.mamba(multimodal_features)
        # 生成最终的logits
        logits = self.head(mamba_features)

        return logits


def create_model(config, num_classes, in_features: Dict[str, int]):  # config = config.models
    models = []

    for model_name in config.names:                 # 遍历config中models中的names属性：cell_gnn、patch_lv0_gnn等
        model_config = getattr(config, model_name)        # 读取对应的models.names的对应参数
        if model_name.lower().startswith(CELL_GNN_MODEL):
            model = GNN(
                    prefix=model_name,
                    model_name=model_config.model_name,    # gcntopk2
                    in_features=in_features[CELL_GRAPH],
                    num_classes=0,
                    hidden_features=model_config.hidden_features,
                    out_features=model_config.out_features,
                    pooling=model_config.pooling,
                    activation=model_config.activation,
            )
            models.append(model)
        elif model_name.lower().startswith(PATCH_LV0_GNN_MODEL):
            model = GNN(
                    prefix=model_name,
                    model_name=model_config.model_name,    # gcntopk2
                    in_features=in_features[PATCH_GRAPH_LV0],
                    num_classes=0,
                    hidden_features=model_config.hidden_features,
                    out_features=model_config.out_features,
                    pooling=model_config.pooling,
                    activation=model_config.activation,
            )
            models.append(model)

        elif model_name.lower().startswith(PATCH_LV1_GNN_MODEL):
            model = GNN(
                    prefix=model_name,
                    model_name=model_config.model_name,    # gcntopk2
                    in_features=in_features[PATCH_GRAPH_LV1],
                    num_classes=0,
                    hidden_features=model_config.hidden_features,
                    out_features=model_config.out_features,
                    pooling=model_config.pooling,
                    activation=model_config.activation,
            )
            models.append(model)

        elif model_name.lower().startswith(TISSUE_GNN_MODEL):
            model = GNN(
                    prefix=model_name,
                    model_name=model_config.model_name,    # gcntopk2
                    in_features=in_features[TISSUE_GRAPH],
                    num_classes=0,
                    hidden_features=model_config.hidden_features,
                    out_features=model_config.out_features,
                    pooling=model_config.pooling,
                    activation=model_config.activation,
            )
            models.append(model)

        elif model_name.lower().startswith(FUSION_MLP):  # FUSION_MLP = 'fusion_mlp'
            fusion_model = functools.partial(
                FusionMLP,
                prefix=model_name,
                num_classes=num_classes,
                hidden_features=model_config.hidden_features,
                adapt_in_features=model_config.adapt_in_features,
                activation=model_config.activation,
                dropout_prob=model_config.dropout_prob,
                normalization=model_config.normalization,
            )
        elif model_name.lower().startswith(FUSION_TRANSFORMER):  # FUSION_TRANSFORMER = 'fusion_transformer'
            fusion_model = functools.partial(
                FusionTransformer,
                prefix=model_name,
                num_classes=num_classes,
                hidden_features=model_config.hidden_features,
                adapt_in_features=model_config.adapt_in_features,
            )

        elif model_name.lower().startswith(FUSION_MAMBA):  # FUSION_MAMBA = 'fusion_mamba'
            fusion_model = functools.partial(
                FusionMamba,
                prefix=model_name,
                num_classes=num_classes,
                hidden_features=model_config.hidden_features,
                adapt_in_features=model_config.adapt_in_features,  # 假设配置中有适配输入特征的策略
            )

        else:
            raise ValueError(f"unknown model name: {model_name}")
    return fusion_model(models=models)

