# 功能：主文件
# 作者：ljc
# 时间：2018.3.6

import argparse, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

configure("loger/MSRN")
import time
# 导入数据处理模块
from data import DatasetFromHdf5
# 导入网络模型
from model import MSRN

# 参数设置
parser = argparse.ArgumentParser(description="SR model")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=20, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-8, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--dataset", default="/disk/FSR_data/train_4_64_big.h5", type=str, help="path to load dataset")
parser.add_argument("--number", default="1", type=int, help="path to load dataset")
parser.add_argument("--scale", default=4, type=int, help="upscale rate to train network")
parser.add_argument("--blocks", default=18, type=int, help="num of MSRBlocks, default: 18")


# 训练函数
def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        # 获取数据
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        # 启用GPU
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        # 调用模型进行图像重建
        sr = model(input)
        # 计算重建出的SR图像与原图之间的loss
        loss = criterion(sr, label)

        # 启用优化器并将梯度缓存清空，再进行反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        if iteration % 1000 == 0:
            number = opt.number
            save_checkpoint_iter(model, number)
            opt.number += 1


# 调整学习率
def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


# 模型及参数保存
def save_checkpoint(model, epoch):
    model_folder = "train/MSRN/"
    model_out_path = model_folder + "{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# 次模型及参数保存
def save_checkpoint_iter(model, number):
    model_folder = "train/MSRN_iter/"
    model_out_path = model_folder + "{}.pth".format(number)
    state = {"epoch": number, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# 主函数
def main():
    # 定义全局变量，加载所有参数并打印
    global opt, model
    opt = parser.parse_args()
    print(opt)

    # 是否使用GPU判断
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # 设置随机种子 random.randint(1, 10000)
    opt.seed = 8787
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    # 加载数据集
    print("===> Loading datasets")
    train_set = DatasetFromHdf5(opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    # 构建模型及损失函数
    print("===> Building model and loss function")
    model = MSRN(opt.scale, opt.blocks)
    criterion = nn.L1Loss(size_average=True)

    # 设置GPU
    print("===> Setting GPU")
    if cuda:
        # 调用多个GPU
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        # 调用单个GPU model = model.cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    # 加载预预训练好的模型及权重
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    # 设置优化器
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 启动训练函数并保存网络模型及参数
    print("===> Training")
    # 开始计时
    start_time = time.time()
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)
    end_time = time.time()
    use_time = end_time - start_time
    print("takes '{}' seconds".format(use_time))


if __name__ == "__main__":
    main()
