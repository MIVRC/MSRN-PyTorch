# 功能：测试文件
# 作者：ljc
# 时间：2017.12.19

import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math 
import scipy.io as sio
from os import listdir
from os.path import join
from skimage.measure import compare_ssim

parser = argparse.ArgumentParser(description="SR test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="train/MSRN_iter/348.pth", type=str, help="model path")
parser.add_argument("--imagepath", default="/disk/test_dataset/set5_x4", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="")

opt = parser.parse_args()
cuda = opt.cuda

# PSNR 计算函数
# 为了测试公平，去除图像周边scale大小的像素点。如，放大2倍，则在图像四周都去掉2个像素点的边缘
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

# SSIM 计算函数
def SSIM(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    ssim = compare_ssim(gt, pred, data_range=gt.max() - pred.min())
    if ssim == 0:
        return 100
    return ssim

# 判断图像是否为.mat文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

# 加载训练好的模型和权重
# "model"中的model为之前训练保存的类型
model = torch.load(opt.model)["model"]

# 判断是否启用GPU
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# 加载所有待测试的图像
image = [join(opt.imagepath, x) for x in listdir(opt.imagepath) if is_image_file(x)]

# sum_bicubic_psnr: bicubic的PSNR测试结果  sum_predicted_psnr：重建后的图像的PSNR测试结果
# sum_bicubic_ssim：bicubic的SSIM测试结果  sum_predicted_ssim：重建后的图像的SSIM测试结果
# all_use_time：重建花的总时间
sum_bicubic_psnr = 0
sum_predicted_psnr = 0
sum_bicubic_ssim = 0
sum_predicted_ssim = 0
all_use_time = 0

# 测试
for _, j in enumerate(image):
    print(j)

    # 所有图像只进行Y通道的测试，在测试图像预处理的时候已经只保留了Y通道的数据
    # im_gt_y：待对比的label图像
    # im_b_y：bicubic放大后的图像
    # im_l_y：输入的LR图像
    im_gt_y = sio.loadmat(j)['im_gt_y']
    im_b_y = sio.loadmat(j)['im_b_y']
    im_l_y = sio.loadmat(j)['im_l_y']

    im_gt_y = im_gt_y.astype('double')
    im_b_y = im_b_y.astype('double')
    im_l_y = im_l_y.astype('double')

    # bicubic法的PSNR和SSIM测量，作为基准对比数据
    psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)
    sum_bicubic_psnr += psnr_bicubic
    ssim_bicubic = SSIM(im_gt_y, im_b_y, shave_border=opt.scale)
    sum_bicubic_ssim += ssim_bicubic

    # 重建图像测试
    # 由于模型在训练的时候采用的是0-1的数据范围，而测试数据采用的是0-255的数据范围，所以在送入模型前先做一个转换
    # .view() 返回一个新的张量与原来张量数据相同但大小不同； -1 代表该维度的大小由其他维度自动推导得出
    im_input = im_l_y / 255.
    im_input = Variable(torch.from_numpy(im_input).float(), volatile=True).view(1, -1, im_input.shape[0], im_input.shape[1]).cuda()

    # 调用GPU
    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    # model(im_input)

    # 开始计时
    start_time = time.time()
    # 调用训练好的模型
    SR = model(im_input)
    end_time = time.time()
    use_time = end_time - start_time
    all_use_time += use_time

    # 重建图像后再转为cpu上进行PSNR和SSIM的计算，因为这两个函数是自己写的，没有实现GPU计算
    SR = SR.cpu()
    im_sr_y = SR.data[0].numpy().astype(float)

    # 将数据范围转为 0-255
    im_sr_y = im_sr_y * 255.
    im_sr_y[im_sr_y < 0] = 0
    im_sr_y[im_sr_y > 255.] = 255.
    im_sr_y = im_sr_y[0, :, :]

    # 重建图像的PSNR和SSIM测量，作为最后的实验数据
    psnr_predicted = PSNR(im_gt_y, im_sr_y, shave_border=opt.scale)
    sum_predicted_psnr += psnr_predicted
    ssim_predicted = SSIM(im_gt_y, im_sr_y, shave_border=opt.scale)
    sum_predicted_ssim += ssim_predicted

    print("PSNR_bicubic =", psnr_bicubic)
    print("PSNR_predicted =", psnr_predicted)
    print("SSI_bicubic =", ssim_bicubic)
    print("SSIM_predicted =", ssim_predicted)
    print("User_time =", use_time)

print("Avg bicubic psnr =", sum_bicubic_psnr / len(image))
print("Avg predicted psnr =", sum_predicted_psnr / len(image))
print("Avg bicubic ssim =", sum_bicubic_ssim / len(image))
print("Avg predicted ssim =", sum_predicted_ssim / len(image))
print("Avg use time =", all_use_time / len(image))
