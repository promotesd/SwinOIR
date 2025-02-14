import os
import math
from decimal import Decimal
import time
import imageio
import torch
import torch.nn.utils as utils
from tqdm import tqdm

import utility  # 假设你有个 utility.py, 提供 make_optimizer, calc_psnr 等

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale  # 比如 [2, 3, 4] 或者仅 [4]

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        # 构造优化器
        self.optimizer = utility.make_optimizer(args, self.model)

        # 如果有指定的预训练/加载点，则恢复优化器状态
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        """
        单个 epoch 的训练流程。
        如果想跑多个 epoch，需要在外部 (main.py) 写循环: 
            for e in range(args.epochs):
                trainer.train()
                trainer.test()
        """
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        # 在日志中记录当前 epoch 和学习率
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        # loss 开始记一行新的日志
        self.loss.start_log()

        # 切换到训练模式
        self.model.train()

        # 准备一些计时器 (可选)
        timer_data = utility.timer()
        timer_model = utility.timer()

        # 如果是多尺度训练，可能需要对每个 scale 都过一遍训练集
        # 如果你只训练单尺度，可直接去掉外层 for idx_scale, scale in enumerate(self.scale)
        for idx_scale, scale in enumerate(self.scale):
            # 告诉数据集当前 scale
            self.loader_train.dataset.set_scale(idx_scale)

            # 遍历训练集中所有 batch
            for batch, (lr, hr, _) in enumerate(self.loader_train):
                # 1) 数据准备阶段
                lr, hr = self.prepare(lr, hr)
                timer_data.hold()  # 计时：数据准备结束
                timer_model.tic()  # 计时：模型开始

                # 2) 前向 + 反向
                self.optimizer.zero_grad()
                sr = self.model(lr, idx_scale)  # 模型推理
                loss = self.loss(sr, hr)        # 计算 loss
                loss.backward()
                self.optimizer.step()           # 优化器更新参数

                # 3) 打印日志
                timer_model.hold()
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        '[{}/{}]\t{}\tLoss: {:.4f} [data: {:.2f}s, model: {:.2f}s]'.format(
                            (batch + 1),
                            len(self.loader_train),
                            self.loss.display_loss(batch),
                            loss.item(),
                            timer_data.release(),
                            timer_model.release()
                        )
                    )
                timer_data.tic()  # 重新开始计时(下个batch的数据加载)

        # 4) 一个 epoch 结束后收尾
        self.loss.end_log(len(self.loader_train))
        # 记录本 epoch 的最终 loss，以便后面 skip_threshold 或别的用途
        self.error_last = self.loss.log[-1, -1]

        # 学习率衰减 (若使用 step/plateau 等)
        self.optimizer.schedule()

    def test(self):
        """
        测试流程：对 self.loader_test 中所有数据集进行推理，并计算 PSNR。
        若不是 test_only，会在最后保存当前 epoch 的模型等。
        """
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        # 为 log 分配一行空间: [1, #testset, #scale]
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))

        self.model.eval()

        timer_test1 = time.perf_counter()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                # 告诉测试集用哪个scale
                d.dataset.set_scale(idx_scale)

                # 遍历测试集的 batch (一般 batch=1)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)          # forward
                    sr = utility.quantize(sr, self.args.rgb_range)

                    # 计算 PSNR
                    psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += psnr

                    # 如果需要保存图像
                    if self.args.save_gt:
                        save_list = [sr, lr, hr]
                        postfix = ('SR', 'LR', 'HR')
                        for v, p in zip(save_list, postfix):
                            normalized = v[0].mul(255 / self.args.rgb_range)
                            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu().numpy()
                            # 示例写法，自己根据路径修改
                            save_path = os.path.join(
                                r'F:\兼职项目\SwinOIR\SwinOIR-main\swinOIR\experiment',
                                r'test',
                                r'results-{}'.format(d.dataset.name),
                                r'{}_x{}_epoch{}_{}.png'.format(filename[0], scale, epoch, p)
                            )
                            imageio.imwrite(save_path, tensor_cpu)

                # 计算该数据集此 scale 的平均 PSNR
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        timer_test2 = time.perf_counter()
        self.ckp.write_log('Forward: {:.2f}s'.format(timer_test2 - timer_test1))
        self.ckp.write_log('Saving...')

        # 如果不是仅测试，就保存模型
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test2 - timer_test1), refresh=True
        )
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        """
        把数据 move 到 GPU (或 CPU)，并且若 precision=half 则转 half。
        """
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        """
        在 main.py 中常用来判断是否结束整个训练流程
        """
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
