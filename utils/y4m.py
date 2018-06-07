import numpy as np
from skimage.transform import resize
from skimage.util.dtype import convert 

class Y4MFile:
    def __init__(self, fname):
        self.fd = open(fname, 'rb')
        if self.fd is None:
            raise Exception("Invalid file")

        magic = b"YUV4MPEG2"
        if self.fd.read(len(magic)) != magic:
            raise Exception("Not Y4M File")
        strm_header = self.fd.readline()
        hdr_fields = strm_header.split()
        self.width = None
        self.height = None
        for field in hdr_fields:
            if field[0] == ord(b'W'):
                self.width = int(field[1:])
            elif field[0] == ord(b'H'):
                self.height = int(field[1:])
            elif field[0] == ord(b'C'):
                assert(False)
            elif field[0] == ord(b'I') and field[1] != ord(b'p'):
                assert(False)

        if self.width is None or self.height is None:
            raise Exception("Height / Width not specified")

        self.halfwidth = self.width // 2
        self.halfheight = self.height // 2

    def next_frame(self):
        hdr = self.fd.readline()
        if hdr == b"":
            return []

        if hdr[0:5] != b"FRAME":
            raise Exception("Invalid Y4M Frame")

        raw_y = self.fd.read(self.width*self.height)
        raw_u = self.fd.read(self.halfwidth*self.halfheight)
        raw_v = self.fd.read(self.halfwidth*self.halfheight)

        y = convert(np.frombuffer(raw_y, dtype='uint8').reshape(self.height, self.width), np.float32)
        u = resize(convert(np.frombuffer(raw_u, dtype='uint8').reshape(self.halfheight, self.halfwidth), np.float32), (self.height, self.width), 0, 'reflect')
        v = resize(convert(np.frombuffer(raw_v, dtype='uint8').reshape(self.halfheight, self.halfwidth), np.float32), (self.height, self.width), 0, 'reflect')

        yuv = np.array([y, u, v])
        from skimage.color import yuv2rgb
        rgb = yuv2rgb(yuv.transpose(1, 2, 0)).transpose(2, 0, 1)
        #transform = np.array([[1, 0, 1.139883], [1, -0.39464233, -0.580621850], [1, 2.03206, 0]], dtype=np.float32)

        #rgb = yuv.transpose(1, 2, 0).dot(transform).transpose(2, 0, 1)
        return rgb
