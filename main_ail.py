import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SRNet import acSRNet
from dataset import DatasetFromHdf5
import numpy as np
import copy
from MyLoss import initLoss, ailLoss
import warnings
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description="Train lightweight vdsr model with AIL")
parser.add_argument("--round", type=int, default=10, help="Round of adaptive importance learning (AIL)")
parser.add_argument("--width", type=int, default=13, help="width of the lightweight network, num of feature maps")
parser.add_argument("--scale", type=int, default=2, help="SR scales, e.g., 2,3,4")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train the SR model in one AIL round")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default=None, type=int, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--tea', default='model/tea_vdsr.pth', type=str, help='path to the teacher model')
parser.add_argument('--premodel', default='model/pre_vdsr_f13.pth', type=str, help='path to pre-trained lightweight model with the traditional learning scheme')
parser.add_argument('--dataset', default='./data/train_291_2.h5', type=str, help='path to training data in .h5 file')


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
    tea_model = torch.load(opt.tea)["model"]
    pre_model = torch.load(opt.premodel)["model"]

    Cn = opt.width
    model = acSRNet(Cn)

    # with the pre-trained model, the training of AIL will be more stable
    model.input.weight.data = copy.deepcopy(pre_model.module.input.weight.data)
    model.output.weight.data = copy.deepcopy(pre_model.module.output.weight.data)
    model.residual_layer.load_state_dict(pre_model.module.residual_layer.state_dict())

    criterion_init = initLoss(size_average=False)
    criterion_ail = ailLoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion_init = criterion_init.cuda()
        criterion_ail = criterion_ail.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        model_resume = "model/" + "ours_ail_r{}_f{}_s{}.pth".format(opt.resume - 1, opt.width, opt.scale)
        if os.path.isfile(model_resume):
            print("=> loading checkpoint '{}'".format(model_resume))
            checkpoint = torch.load(model_resume)
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            exit(0)

    print("===> Setting Optimizer")
    optimizer = optim.SGD([
        {'params': model.module.parameters()}
    ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    lossAarry = np.zeros(opt.nEpochs * opt.round)

    num = 0
    if not opt.resume:
        for curr in range(opt.round):
            reset_learning_rate(optimizer)
            if curr < 1:
                # for epoch in range(opt.start_epoch, opt.nEpochs + 1):
                pbar = tqdm(range(opt.start_epoch, opt.nEpochs + 1))
                for epoch in pbar:
                    lossAarry[num] = train_init(training_data_loader, optimizer, model, tea_model, criterion_init, epoch)
                    pbar.set_description("loss: %.8f" % (lossAarry[num]))
                    pbar.update()
                    num = num + 1
                pbar.close()

            else:
                pre_learned = copy.deepcopy(model)
                #for epoch in range(opt.start_epoch, opt.nEpochs + 1):
                pbar = tqdm(range(opt.start_epoch, opt.nEpochs + 1))
                for epoch in pbar:
                    lossAarry[num] = train_ail(training_data_loader, optimizer, model, tea_model,  criterion_ail,
                                               epoch, pre_learned, curr)
                    pbar.set_description("loss: %.8f" % (lossAarry[num]))
                    pbar.update()
                    num = num + 1

                pbar.close()

            save_checkpoint(model, curr)
    else:
        for curr in range(opt.round - opt.resume):
            reset_learning_rate(optimizer)
            pre_learned = copy.deepcopy(model)
            # for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            pbar = tqdm(range(opt.start_epoch, opt.nEpochs + 1))
            for epoch in pbar:
                lossAarry[num] = train_ail(training_data_loader, optimizer, model, tea_model, criterion_ail, epoch,
                                           pre_learned, curr + opt.resume)
                pbar.set_description("loss: %.8f" % (lossAarry[num]))
                pbar.update()
                num = num + 1

            pbar.close()

            save_checkpoint(model, curr + opt.resume)


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))#0.2
    return lr


def reset_learning_rate(optimizer):
    """reset learning rate to the original """
    for param_group in optimizer.param_groups:
        param_group["lr"] = opt.lr


def train_init(training_data_loader, optimizer, model, tea_model, criterion, epoch):
    # training the model with initialized importance weight generated by the teacher model

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

        outS = model(input)
        outT = tea_model(input)
        outT = outT.detach()
        outT.requires_grad = False
        loss = criterion(outS, outT, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data

    return lossValue


def train_ail(training_data_loader, optimizer, model, tea_model, criterion, epoch, pre_learned, curr):
    # training the model with adaptively updated importance weight

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

        outS = model(input)
        outT = tea_model(input)
        outT = outT.detach()
        outT.requires_grad = False

        outS_p = pre_learned(input)
        outS_p = outS_p.detach()
        outS_p.requires_grad = False

        loss = criterion(outS, outS_p, outT, target, curr - 1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data

    return lossValue


def save_checkpoint(model, epoch):
    model_out_path = "model/" + "ours_ail_r{}_f{}_s{}.pth".format(epoch, opt.width, opt.scale)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint in round {} has been saved to {}".format(opt.round, model_out_path))


if __name__ == "__main__":
    main()
    exit(0)
