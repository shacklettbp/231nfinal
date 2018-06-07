from .file import VideoFile
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchnet.dataset import ListDataset
from torchnet.dataset.dataset import Dataset
from torchvision.transforms import ToTensor
import torch
import sys
import math
import numpy as np

def load_func(line):
    f = VideoFile(line)
    return f

def batchify(batch):
    conv = ToTensor()
    batch_list = []
    for f in batch:
        video = []
        for i in range(f.length()):
            frame = f.next_frame()
            video.append(conv(frame))

        video = torch.stack(video)
        _, _, height, width = video.shape
        new_height = math.ceil(height / 32)*32
        new_width = math.ceil(width / 32)*32
        pad_height = new_height - height
        pad_width = new_width - width
        video = F.pad(video, (32, pad_width + 32, 32, pad_height + 32), 'constant', 0)

        batch_list.append(video)

    return batch_list

class VideoDataset(Dataset):
    def __init__(self, list_file, train_len):
        super(VideoDataset, self).__init__()

        if isinstance(list_file, str):
            with open(list_file) as f:
                vid_list = [line.replace('\n', '').split() for line in f]
                vid_list = [(e[0], int(e[1])) for e in vid_list]

        self.list = []
        for video, length in vid_list:
            offset = 0
            while length >= train_len:
                self.list.append((video, offset, train_len))
                length -= train_len
                offset += train_len
                if length < train_len and length > 0:
                    offset -= train_len - length
                    length = train_len

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        super(VideoDataset, self).__getitem__(idx)

        return VideoFile(*self.list[idx])

def load_videos(list_file, chunk_size=30, batch_size=4):
    dataset = VideoDataset(list_file, chunk_size)
    dataset = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, collate_fn=batchify, shuffle=True) #This will load data when needed, in parallel, up to <num_workers> thread.

    return dataset
