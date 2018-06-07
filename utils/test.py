import cv2

cap = cv2.VideoCapture("../seqs/random/akiyo_cif.y4m")
ret, frame = cap.read()
assert(ret)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(frame.shape)

from PIL import Image

img = Image.fromarray(frame)
img.save("/tmp/t.png")
