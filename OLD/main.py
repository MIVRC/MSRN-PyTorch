import argparse, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DatasetFromHdf5
from model import MSRN

parser = argparse.ArgumentParser(description="PyTorch MSRN train.py")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=20, help="number of epochs to for train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-8, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--dataset", default="data_x2.h5", type=str, help="path to load dataset")
parser.add_argument("--number", default="1", type=int, help="path to load dataset")

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])


    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad = False)

        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        sr = model(input)
        loss = criterion(sr, label)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        if iteration % 10 == 0:
            log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),loss.data[0]))


def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def save_checkpoint(model, epoch):
    model_folder = "Weights/"
    model_out_path = model_folder + "{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = 8787
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5(opt.dataset)
    training_data_loader = DataLoader(dataset = train_set, num_workers = opt.threads, batch_size = opt.batchSize, shuffle = True)

    print("===> Building model and loss function")
    model = MSRN()
    criterion = nn.L1Loss(size_average = True)

    print("===> Setting GPU")
    if cuda:
        model = nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

if __name__ == "__main__":
    main()
