import cv2, os, pickle, numpy as np, torch
from tqdm import tqdm
from rembg.bg import remove
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .common import size, batch, arr_size
from .read import initial_rang

class Remove:
    def __init__(self, dirname):
        self.dirname = dirname
        self.size = (size, size)

    def dump(self):
        for filename in os.listdir(self.dirname):
            filename = filename.replace('.mp4', '')
            print('removing '+ f'{self.dirname}/{filename}.mp4')
            self.read(filename)

    def compare(self):
        for pkl in os.listdir('out/src/removed'):
            with open('out/src/edited/'+pkl, 'rb') as f:
                edited = pickle.load(f)
            with open('out/src/removed/'+pkl, 'rb') as f:
                removed = pickle.load(f)
            match = np.where((edited==255)&(removed==255),1,0).sum()/initial_rang

            print(f'{pkl.replace(".pkl","")} percent: {match*100/len(edited):>0.2f}%')

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
            writer.write(cv2.merge(bin[0]))
            arr = bin if arr.size == 0 else np.append(arr, bin, axis=0)

        with open(pkl, 'wb') as f:
            pickle.dump(arr, f)

class Cam:
    def __init__(self, filename, cam, cl=None):
        ansmap = { 'elegant': 0, 'dance': 1 }
        self.filename = filename.replace('.mp4', '')
        cl = cl if cl else ansmap.get(self.filename.split('_')[-1], 2)
        self.targets = [ClassifierOutputTarget(cl) for _ in range(batch)]

        with open(f'out/src/edited/{self.filename}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.cam = cam

    def dump(self):
        print('grad cam '+ f'{self.filename}.mp4')
        edited = f'out/video/edited/{self.filename}.mp4'
        cam = f'out/video/cam/{self.filename}.mp4'

        cap = cv2.VideoCapture(edited)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(cam, fmt, fps, (size, size), 0)

        imgs = np.zeros((frame_count, size, size))
        for fr in tqdm(range(arr_size, frame_count)):
            if fr < arr_size*2:
                mean = fr-arr_size+1
            elif frame_count-arr_size < fr:
                mean = frame_count-fr
            else:
                mean = arr_size
            img = self.run(fr)
            imgs[fr-arr_size:fr] += img
            writer.write((imgs[fr-arr_size]*255/mean).astype(np.uint8))

    def run(self, fr):
        idx = np.array(list(map(lambda _:np.arange(fr-arr_size,fr),range(batch))))
        input_tensor = torch.Tensor(self.data[idx])
        return self.cam(input_tensor, self.targets)