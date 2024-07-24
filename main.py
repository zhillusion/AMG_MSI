from trainer import MMTrainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str,
                        default='example.yaml')
    args = parser.parse_args()
    # 解析命令行输入的参数 可通过 args 对象访问
    mmtrainer = MMTrainer(args.config_path)
    print('start training')
    mmtrainer.train()

    # 用MMTrainer实例的train方法开始训练
