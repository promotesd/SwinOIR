from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# 这个简单封装是供多数据集拼接使用
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        # 这里假设每个 dataset 都有 .train 属性
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'):
                d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        """
        同时构造 self.loader_train 和 self.loader_test:
          1) self.loader_train: 若不是 test_only，就加载 args.data_train 指定的数据集
          2) self.loader_test: 加载 args.data_test 指定的数据集
        """
        ################################################################
        # 1. 构建训练集 (self.loader_train)
        ################################################################
        self.loader_train = None

        # 只有当不是 test_only 时，才构建训练集
        if not args.test_only:
            train_datasets = []

            # 假设你在命令行或配置里加了 --data_train=Set5+Set14 之类
            # 可以一次加载多个数据集
            for d in args.data_train.split('+'):
                # 如果数据集名称在这几个里，就用 Benchmark 类，否则用 SRData
                if d in ['Set5', 'Set14', 'B100', 'Urban100', 'PIRM', 'manga109']:
                    m = import_module('data.benchmark')
                    # 注意: 这里指定 benchmark=False，让它当“训练集”用
                    # 同时 Benchmark 里 _set_filesystem 会去 dir_data/benchmark/ 下面找
                    trainset = getattr(m, 'Benchmark')(args, name=d)
                    trainset.train = True  # 强行声明下自己是 train
                else:
                    m = import_module('data.srdata')
                    trainset = getattr(m, 'SRData')(args, name=d, benchmark=False)

                train_datasets.append(trainset)

            # 如果有多个训练数据集，就拼在一起
            if len(train_datasets) > 1:
                trainset = MyConcatDataset(train_datasets)
            elif len(train_datasets) == 1:
                trainset = train_datasets[0]
            else:
                trainset = None

            if trainset is not None:
                self.loader_train = dataloader.DataLoader(
                    trainset,
                    batch_size=getattr(args, 'batch_size', 16),  # 可自行修改
                    shuffle=True,      # 训练集一般要 shuffle
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads
                )

        ################################################################
        # 2. 构建测试集 (self.loader_test)
        ################################################################
        self.loader_test = []

        # 如果 args.data_test = "Set5+Set14" 会被 split 成 ["Set5", "Set14"]
        for d in args.data_test:
            # 这几个是常见 benchmark 数据集
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'PIRM', 'manga109']:
                m = import_module('data.benchmark')
                # benchmark=True，说明是测试集
                testset = getattr(m, 'Benchmark')(args, name=d)
            else:
                m = import_module('data.srdata')
                testset = getattr(m, 'SRData')(args, name=d, benchmark=True)

            # 测试集一般 batch_size=1, shuffle=False
            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads
                )
            )
