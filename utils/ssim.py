import math
import torch
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor 

def make_kernel(size, sigma=1.5):
    coords = torch.range(0, size - 1).type(dtype)
    coords -= (size - 1) / 2.0

    g = coords**2
    g *= -0.5 / sigma**2

    g = g.view(1, -1) + g.view(-1, 1)
    g = g.view(1, -1)
    g = F.softmax(g, dim=1)

    return g.view(1, 1, size, size)

kernels = {}
kernels[(11, 3)] = make_kernel(11).expand(3, -1, -1, -1).contiguous()
kernels[(8, 3)] = make_kernel(8).expand(3, -1, -1, -1).contiguous()
kernels[(6, 3)] = make_kernel(6).expand(3, -1, -1, -1).contiguous()
kernels[(3, 3)] = make_kernel(3).expand(3, -1, -1, -1).contiguous()
kernels[(4, 3)] = make_kernel(4).expand(3, -1, -1, -1).contiguous()
kernels[(2, 3)] = make_kernel(2).expand(3, -1, -1, -1).contiguous()

def ssim_helper(x, y, reducer, max_val, compensation):
    SSIM_K1 = 0.01
    SSIM_K2 = 0.03
    c1 = (SSIM_K1 * max_val)**2
    c2 = (SSIM_K2 * max_val)**2

    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = mean0**2 + mean1**2
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(x * y) * 2.0
    den1 = reducer(x**2 + y**2)
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    return luminance, cs

def ssim_per_channel(img1, img2, filter_size):
    channels = img1.shape[1]
    filter_size = min(img1.shape[2], filter_size)
    kernel = kernels[(filter_size, channels)]
    
    compensation = 1.0
    def reducer(x):
        shape = x.shape
        x = x.view(-1, *shape[-3:])
        y = F.conv2d(x, kernel, stride=1, padding=filter_size//2, groups=channels)
        return y.view(*shape[:-3], *y.shape[1:])

    luminance, cs = ssim_helper(img1, img2, reducer, 1.0, compensation)
    prod = luminance * cs
    ssim_val = torch.mean(prod.view(-1, channels, prod.shape[2]*prod.shape[3]), dim=2)
    cs = torch.mean(cs.view(-1, channels, cs.shape[2]*cs.shape[3]), dim=2)
    return ssim_val, cs

def ssim(img1, img2, filter_size=11):
    per_channel, cs = ssim_per_channel(img1, img2, filter_size)
    return torch.mean(torch.mean(per_channel, 1), 0), torch.mean(torch.mean(cs, 1), 0)

def msssim(img1, img2):
    MSSSIM_WEIGHTS = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).type(dtype)
    msssim = []
    mcs = []
    for i in range(MSSSIM_WEIGHTS.shape[0]):
        sim, cs = ssim(img1, img2, filter_size=11)
        msssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    msssim = torch.stack(msssim)
    mcs = torch.stack(mcs)
    return torch.prod((mcs ** MSSSIM_WEIGHTS) * (msssim ** MSSSIM_WEIGHTS))

