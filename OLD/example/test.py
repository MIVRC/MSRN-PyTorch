import argparse
import torch
from torch.autograd import Variable
import numpy as np
import math
import scipy.io as sio
from os import listdir
from os.path import join
from skimage.measure import compare_ssim

parser = argparse.ArgumentParser(description="PyTorch MSRN test.py")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="final_x3.pth", type=str, help="model path")
parser.add_argument("--imagepath", default="../../PSNR_test/3/set5_x3", type=str, help="image path")
parser.add_argument("--scale", default=3, type=int, help="")
opt = parser.parse_args()

cuda = opt.cuda

# PSNR
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

# SSIM
def SSIM(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    ssim = compare_ssim(gt, pred, data_range=gt.max() - pred.min())
    if ssim == 0:
        return 100
    return ssim

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

model = torch.load(opt.model)["model"]

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

image = [join(opt.imagepath, x) for x in listdir(opt.imagepath) if is_image_file(x)]

sum_bicubic_psnr = 0
sum_predicted_psnr = 0
sum_bicubic_ssim = 0
sum_predicted_ssim = 0

for _, j in enumerate(image):
    print(j)

    im_gt_y = sio.loadmat(j)['im_gt_y']
    im_b_y = sio.loadmat(j)['im_b_y']
    im_l_y = sio.loadmat(j)['im_l_y']

    im_gt_y = im_gt_y.astype('double')
    im_b_y = im_b_y.astype('double')
    im_l_y = im_l_y.astype('double')

    psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)
    sum_bicubic_psnr += psnr_bicubic
    ssim_bicubic = SSIM(im_gt_y, im_b_y, shave_border=opt.scale)
    sum_bicubic_ssim += ssim_bicubic

    im_input = im_l_y / 255.
    im_input = Variable(torch.from_numpy(im_input).float(), volatile=True).view(1, -1, im_input.shape[0], im_input.shape[1]).cuda()

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    SR = model(im_input)

    SR = SR.cpu()
    im_sr_y = SR.data[0].numpy().astype(float)

    im_sr_y = im_sr_y * 255.
    im_sr_y[im_sr_y < 0] = 0
    im_sr_y[im_sr_y > 255.] = 255.
    im_sr_y = im_sr_y[0, :, :]

    psnr_predicted = PSNR(im_gt_y, im_sr_y, shave_border=opt.scale)
    sum_predicted_psnr += psnr_predicted
    ssim_predicted = SSIM(im_gt_y, im_sr_y, shave_border=opt.scale)
    sum_predicted_ssim += ssim_predicted

    print("PSNR_bicubic =", psnr_bicubic)
    print("PSNR_predicted =", psnr_predicted)
    print("SSI_bicubic =", ssim_bicubic)
    print("SSIM_predicted =", ssim_predicted)

print("Avg bicubic psnr =", sum_bicubic_psnr / len(image))
print("Avg predicted psnr =", sum_predicted_psnr / len(image))
print("Avg bicubic ssim =", sum_bicubic_ssim / len(image))
print("Avg predicted ssim =", sum_predicted_ssim / len(image))
