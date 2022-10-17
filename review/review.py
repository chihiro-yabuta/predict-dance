import cv2, os, pickle, numpy as np
from tqdm import tqdm
from rembg.bg import remove
from ..src.py.common import size
from ..src.py.read import initial_rang

class Review:
    def __init__(self, dirname):
        self.dirname = dirname
        self.size = (size, size)

    def dump(self):
        for filename in os.listdir(self.dirname):
            filename = filename.replace('.mp4', '')
            print('back removing '+ f'{self.dirname}/{filename}.mp4')
            self.read(filename)

    def compare(self):
        for pkl in os.listdir('out/src/removed'):
            with open('out/src/edited/'+pkl, 'rb') as f:
                edited = pickle.load(f)
            with open('out/src/removed/'+pkl, 'rb') as f:
                removed = pickle.load(f)
            match = np.where((edited==255)&(removed==255),1,0).sum()/initial_rang

            print(f'{pkl.replace(".pkl", "")} match deg: {match*100/len(edited)}%')

    def read(self, filename):
        arr = np.array([])
        video = f'{self.dirname}/{filename}.mp4'
        removed = f'out/video/removed/{filename}.mp4'
        pkl = f'out/src/removed/{filename}.pkl'

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
            bin = np.where(gray == 0, 0, 255).astype(np.uint8)[np.newaxis,np.newaxis,:,:]
            writer.write(cv2.merge(bin))
            arr = bin if arr.size == 0 else np.append(arr, bin, axis=0)

        with open(pkl, 'wb') as f:
            pickle.dump(arr, f)