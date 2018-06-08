import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
from utils.file import VideoFile
from utils.load import load_videos
from utils.ssim import ssim, msssim
import math

import multiprocessing
multiprocessing.set_start_method('spawn', force=True) # OpenCV in dataloader

import matplotlib.pyplot as plt
import gc
import sys

ToImage = T.ToPILImage()

class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(dim=2)
    
class Debug(nn.Module):
    def forward(self, x):
        print("Debug ")
        print(x.shape)
        return x
    
class FullCompress(nn.Module):
    def __init__(self, predictor, autoencoder):
        super(FullCompress, self).__init__()
        self.predictor = predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        
        self.autoencoder = autoencoder
    def forward(self, ctx, truth):
        with torch.no_grad():
            pred = self.predictor(ctx)

        res = truth - pred 
        enc_res = self.autoencoder(res)

        return pred + enc_res

class ResBlock(nn.Module):
    def __init__(self, size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += x
        out = self.relu(out)
        
        return out
    
class ResBlock3d(nn.Module):
    def __init__(self, size):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(size, size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(size, size, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += x
        out = self.relu(out)
        
        return out

dtype = torch.cuda.FloatTensor 

class Quantize(nn.Module):
    def forward(self, x):
        x = F.relu(x) # hack hack hack
        return x
        shape = x.shape
        flat = x.view(-1)
        maximum = torch.max(flat)
        minimum = torch.min(flat)
        flat = 255.0 * (flat - minimum) / (maximum - minimum)
        flat = flat.type(torch.cuda.ByteTensor)
        flat = flat / 64
        self.compressed = flat
        flat = flat * 64
        flat = flat.type(dtype)
        flat = flat*(maximum - minimum)/255.0 + minimum
        flat = flat.view(*shape)
        return flat

pred_model = nn.Sequential(
    nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
    #nn.BatchNorm3d(32),
    nn.ReLU(),
    ResBlock3d(32),
    nn.Conv3d(32, 64, kernel_size=3, stride=(1, 3, 3), padding=(1, 0, 0)), # Reduce to 32x32
    #nn.BatchNorm3d(64),
    nn.ReLU(),
    nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=(0, 1, 1)), # Reduce time dimension
    Squeeze(),
    #nn.BatchNorm2d(64),
    nn.ReLU(),
    ResBlock(64),
    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
).type(dtype)

enc_model = nn.Sequential(
    nn.Conv2d(3, 64, 3, stride=1, padding=1),
    nn.ReLU(),
    ResBlock(64),
    nn.Conv2d(64, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 8, stride=6, padding=2),
    nn.ReLU(),
    nn.Conv2d(64, 8, 1, stride=1, padding=0),
    #nn.ReLU(),
    Quantize(),
    nn.ConvTranspose2d(8, 64, 1, stride=1, padding=0),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 64, 8, stride=6, padding=2),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    ResBlock(64),
    nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
).type(dtype)

"""
enc_model = nn.Sequential(
    nn.Conv2d(3, 64, 3, stride=1, padding=1),
    nn.ReLU(),
    ResBlock(64),
    nn.Conv2d(64, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    #nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, 8, stride=6, padding=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, 2, stride=1, padding=0),
    nn.ReLU(),
    #nn.BatchNorm2d(64),
    nn.Conv2d(64, 4, 1, stride=1, padding=0),
    nn.ReLU(),
    #nn.BatchNorm2d(8),
#    Quantize(),
    nn.ConvTranspose2d(4, 64, 1, stride=1, padding=0),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 64, 2, stride=1, padding=0),
    nn.ReLU(),
    #nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 64, 8, stride=6, padding=2),
    nn.ReLU(),
    #nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    ResBlock(64),
    nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
).type(dtype)
"""

model = FullCompress(pred_model, enc_model)

model.load_state_dict(torch.load("model_backctx_auto"))
model = model.eval()
model = model.type(dtype)

vfile = VideoFile(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))


video = vfile.next_n_tensor(vfile.length())


_, _, height, width = video.shape

output = cv2.VideoWriter("/tmp/o.avi", cv2.VideoWriter_fourcc(*"I420") , 24, (width, height), True)

new_height = math.ceil(height / 32)*32
new_width = math.ceil(width / 32)*32
pad_height = new_height - height
pad_width = new_width - width
video = F.pad(video, (32, pad_width + 32, 32, pad_height + 32), 'constant', 0)

prev_frames = 3

total_height = video.shape[2]
total_width = video.shape[3]
num_chunks_tall = total_height // 32
num_chunks_wide = total_width // 32
pred_frame = torch.zeros(3, total_height, total_width).type(dtype)
ctx = torch.zeros(1, 3, prev_frames, total_height, total_width).type(dtype)
for f in range(video.shape[0]):
    print("Frame {}".format(f))
    for i in range(num_chunks_tall - 2):
        for j in range(num_chunks_wide - 2):
            sub_ctx = ctx[:, :, :, i*32:i*32+96, j*32:j*32+96]
            truth = video[f, :, i*32+32:i*32+64, j*32+32:j*32+64].type(dtype)
            truth = truth.view(1, *truth.shape)
            with torch.no_grad():
                pred = model(sub_ctx, truth)

            pred_frame[:, i*32+32:i*32+64, j*32+32:j*32+64] = pred[0]

    ctx[:, :, 0, :, :] = ctx[:, :, 1, :, :]
    ctx[:, :, 1, :, :] = ctx[:, :, 2, :, :]
    
    if f < prev_frames: # Give ourselves good references at beginning otherwise we never recover
        ctx[0, :, 2, :, :] = video[f, :, :, :]
    else:
        #ctx[0, :, 2, :, :] = pred_frame[:, :, :]
        ctx[0, :, 2, :, :] = video[f, :, :, :]

    if f > prev_frames:
        img = (pred_frame[[2, 1, 0], 32:32+height, 32:32+width].clamp(0, 1)*255).type(torch.ByteTensor).permute(1, 2, 0).numpy()
        output.write(img)


