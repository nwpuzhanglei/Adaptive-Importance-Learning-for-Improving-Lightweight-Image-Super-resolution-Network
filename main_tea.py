import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import DatasetFromHdf5
import numpy as np
import warnings
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description="Train teacher vdsr model in AIL")
parser.add_argument("--scale", type=int, default=2, help="SR scales")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train the SR model in one AIL round")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--start-epoch", type=int, default=1, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", default=1, type=int, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--dataset', default='./data/train_291.h5', type=str, help='path to training data in .h5 file '
                                                                                '(with training data from all scales, '
                                                                                'e.g., 2, 3, 4)')


def main():
    warnings.filterwarnings("ignore")
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5(opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    print("===> Building model")
    # load the pre-trained teacher model and the lightweight model
    model = Net()

    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.SGD([
        {'params': model.module.parameters()}
    ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    num = 0
    lossAarry = np.zeros(opt.nEpochs)
    pbar = tqdm(range(opt.start_epoch, opt.nEpochs + 1))
    for epoch in pbar:
        lossAarry[num] = train(training_data_loader, optimizer, model, criterion, epoch)
        pbar.set_description("loss: %.8f" % (lossAarry[num]))
        pbar.update()
        num = num + 1
    pbar.close()
    save_checkpoint(model)


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))#0.2
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    # training the teacher model in the traditional way

    lr = adjust_learning_rate(epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    model.train()
    lossValue = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data

    return lossValue


def save_checkpoint(model):
    model_out_path = "model/" + "tea_vdsr.pth"
    state = {"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
    print("teacher model has been saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
    exit(0)
