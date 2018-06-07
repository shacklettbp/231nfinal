import cv2
import torch
from torchvision.transforms import ToTensor

class VideoFile:
    def __init__(self, fname, offset, length):
        self.cap = cv2.VideoCapture(fname)
        if not self.cap.isOpened():
            raise Exception("Failed to open {}".format(fname))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        self.len = length

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def next_n_tensor(self, N):
        conv = ToTensor()
        n_frames = []
        for i in range(N):
            n_frames.append(conv(self.next_frame()))

        return torch.stack(n_frames)

    def length(self):
        return self.len

