import argparse, os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torch.nn as nn
from PIL import Image
from os import listdir
from os.path import join
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch MSRN SR.py")
parser.add_argument("--model", default="Weights/final_x2.pth", type=str, help="path to model")
parser.add_argument("--testDir", default="Rebuild/input/Set5/", type=str, help="path to load lr images (we load hr images and directly dowsampling it by the bicubic according the upsampling factors.)")
parser.add_argument("--resultDir", default="Rebuild/out/2/Set5", type=str, help="path to save sr images")
parser.add_argument("--scale", default="2", type=int, help="which upsampling factor")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
opt = parser.parse_args()

cuda = opt.cuda
print(opt)

if not opt.testDir:
    print("TestDir is musted!")
    SystemExit(1)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))

images_path = [join(opt.testDir, x) for x in listdir(opt.testDir) if is_image_file(x)]
print("Load images size: ", len(images_path))

model = torch.load(opt.model)['model']

if cuda:
    model = model.cuda()
else:
    model = model.cpu()

if not os.path.isdir(opt.resultDir):
    os.mkdir(opt.resultDir)

for image_path in tqdm(images_path):
    im_imput = Image.open(image_path).convert('YCbCr')
    h, w = im_imput.size
    im_imput = im_imput.resize((h-h%opt.scale, w-w%opt.scale), Image.BICUBIC)
    filename = image_path.split('/')[-1].split('.')[0]
    y, cb, cr = im_imput.split()
    h, w = y.size
    y = y.resize((h//opt.scale, w//opt.scale), Image.BICUBIC)
    cb = cb.resize((h//opt.scale, w//opt.scale), Image.BICUBIC)
    cr = cr.resize((h//opt.scale, w//opt.scale), Image.BICUBIC)
    
    input = Variable(ToTensor()(y), volatile=True).view(1, -1, y.size[1], y.size[0]).cuda()
    y_sr = model(input)

    y_sr = [y_sr]

    for index, ss in enumerate(y_sr):
        ss = ss.cpu().data[0].numpy()*255.0
        ss = ss.clip(0, 255)
        ss = Image.fromarray(np.uint8(ss[0]), mode='L')
        
        cb_sr = cb.resize(ss.size, Image.BICUBIC)
        cr_sr = cr.resize(ss.size, Image.BICUBIC)

        out_img = Image.merge('YCbCr', [ss, cb_sr, cr_sr]).convert('RGB')
        out_img.save("%s/%s.png"%(opt.resultDir, filename))
