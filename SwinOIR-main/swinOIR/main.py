import torch

import utility
import data
import model
import loss
from option import args       # 假设你在 option.py 中定义了 args
from trainer import Trainer   # 这里的 Trainer 要包含 train() 和 test() 方法

# 设置随机种子，保证可复现
torch.manual_seed(args.seed)

# 创建一个“checkpoint”对象，用来管理日志、模型保存路径等
checkpoint = utility.checkpoint(args)

def main():
    # 只有在 checkpoint.ok = True 时才继续（可能用来检查输出目录是否合法等）
    if checkpoint.ok:
        # 1. 构建 DataLoader (train + test)
        loader = data.Data(args)  # Data类里会生成 loader.loader_train / loader.loader_test

        # 2. 构建模型
        _model = model.Model(args, checkpoint)

        # 3. 如果不是 test_only，就创建 loss 函数；否则不需要
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

        # 4. 创建 Trainer
        t = Trainer(args, loader, _model, _loss, checkpoint)

        # 5. 根据 test_only 决定是否只测试一次或持续训练
        if args.test_only:
            # 仅测试模式：直接调用 test 并退出
            t.test()
        else:
            # 训练模式：只要不终止，就持续 1) 训练 2) 测试
            while not t.terminate():
                t.train()  # 一轮训练
                t.test()   # 测试

        # 训练/测试流程结束后，做一些收尾操作(保存最终模型、写日志等)
        checkpoint.done()

if __name__ == '__main__':
    main()
