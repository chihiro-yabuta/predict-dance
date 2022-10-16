import cv2, os, numpy as np
from tqdm import tqdm
from rembg.bg import remove
from .common import size

class Review:
    def __init__(self, dirname):
        self.dirname = dirname
        self.size = (size, size)

    def dump(self):
        for filename in os.listdir(self.dirname):
            self.read(filename)

    def read(self, filename):
        video = f'{self.dirname}/{filename}'
        removed = f'out/video/removed/{filename}'

        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        h_ma = int((h-w)/2) if w < h else 0
        w_ma = int((w-h)/2) if w > h else 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(removed, fmt, fps, self.size, 0)

        for _ in tqdm(range(frame_count)):
            _, frame = cap.read()
            frame = cv2.resize(remove(frame)[h_ma:h-h_ma, w_ma:w-w_ma], self.size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bin = np.where(gray == 0, 0, 255).astype(np.uint8)
            writer.write(bin)

    def compare(self, filename):
        edited = f'out/video/edited/{filename}'
        removed = f'out/video/removed/{filename}'

        edited_cap = cv2.VideoCapture(edited)
        removed_cap = cv2.VideoCapture(removed)
        frame_count = int(removed_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(frame_count)):
            _, edited_frame = edited_cap.read()
            _, removed_frame = removed_cap.read()