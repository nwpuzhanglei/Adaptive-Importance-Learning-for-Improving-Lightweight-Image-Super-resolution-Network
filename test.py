# -*- coding: utf-8 -*-   produce the results on all datasets with different scaling factor
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import math
import warnings
import scipy.io
import os

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument('--model', default='model/ours_ail_r9_f13_s2.pth', type=str, help='path to our trained model')
parser.add_argument('--premodel', default='model/pre_vdsr_f13.pth', type=str, help='path to pre-trained lightweight model')
parser.add_argument('--tea', default='model/tea_vdsr.pth', type=str, help='path to teacher model')
parser.add_argument("--fpath", default='./', type=str, help="path to the test dataset")
parser.add_argument("--dataset", default="Set5", type=str, help="test set name")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", type=int, default=2, help="SR scales, e.g., 2,3,4")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:, :, 0] = y
    img[:, :, 1] = ycbcr[:, :, 1]
    img[:, :, 2] = ycbcr[:, :, 2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


def SRImage(fpath, setname, imgName, model, scale, mname):

    im_gt_ycbcr = scipy.io.loadmat(fpath + setname + '/' + str(scale) + "/GT/" + imgName + ".mat")
    im_gt_ycbcr = im_gt_ycbcr['im_gt_ycbcr']
    im_b_ycbcr = scipy.io.loadmat(fpath + setname + '/' + str(scale) + "/" + imgName + "_scale.mat")
    im_b_ycbcr = im_b_ycbcr['im_b_ycbcr']

    im_gt_ycbcr = im_gt_ycbcr * 255.0
    im_b_ycbcr = im_b_ycbcr * 255.0

    im_gt_y = im_gt_ycbcr[:, :, 0].astype(float)
    im_b_y = im_b_ycbcr[:, :, 0].astype(float)

    im_input = im_b_y / 255.0

    im_input = Variable(torch.from_numpy(im_input).float(), volatile=True).view(1, -1, im_input.shape[0], im_input.shape[1])

    if cuda:
        model = model.module.cuda()
        im_input = im_input.cuda()

    out = model(im_input)
    out = out.cpu()
    im_h_y = out.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    psnr_predicted = PSNR(im_gt_y, im_h_y[0, :, :], shave_border=scale)

    im_h = colorize(im_h_y[0, :, :], im_b_ycbcr)
    im_gt = Image.fromarray(im_gt_ycbcr.astype(np.uint8), "YCbCr").convert("RGB")
    im_b = Image.fromarray(im_b_ycbcr.astype(np.uint8), "YCbCr").convert("RGB")

    # save the image of SR result as well as the gt and the bicubic input
    save_folder = './result'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, setname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = imgName + '_' + mname + '_' + str(scale)
    save_name_gt = imgName + '_gt_' + str(scale)
    save_name_bicubic = imgName + '_bicubic_' + str(scale)

    im_h.save(os.path.join(save_path, save_name + '.png'))
    if mname == 'AIL':
        im_gt.save(os.path.join(save_path, save_name_gt + '.png'))
        im_b.save(os.path.join(save_path, save_name_bicubic + '.png'))

    # save the brightness channel of the SR result
    scipy.io.savemat(os.path.join(save_path, save_name + '_sr.mat'), {'img': im_h_y[0, :, :]})
    print('PSNR of {} is : {}'.format(mname, psnr_predicted))


opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]
pre_model = torch.load(opt.premodel)["model"]
tea_model = torch.load(opt.tea)["model"]

print('SR result of the image \'{}\' from \'{}\':'.format(opt.image, opt.dataset))
SRImage(opt.fpath, opt.dataset, opt.image, pre_model, opt.scale, 'Lightweight model')
SRImage(opt.fpath, opt.dataset, opt.image, model, opt.scale, 'AIL + lightweight model')
SRImage(opt.fpath, opt.dataset, opt.image, tea_model, opt.scale, 'Teacher')

