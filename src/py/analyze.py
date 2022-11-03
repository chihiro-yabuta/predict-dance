import cv2, os, pickle, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from rembg.bg import remove
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .common import size, batch, arr_size
from .read import initial_rang

class Flow:
    def __init__(self, filename, model):
        self.filename = filename.replace('.mp4', '')
        with open(f'out/src/edited/{self.filename}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.frame, self.model = len(self.data), model

    def read(self):
        print('flow '+ f'{self.filename}.mp4')
        arr = np.array([])
        for fr in tqdm(range(self.frame-arr_size)):
            res = self.forward(fr)
            arr = res if arr.size == 0 else np.append(arr, res, axis=0)
        self.plot(arr)

    def forward(self, fr):
        idx = np.arange(fr,fr+arr_size).repeat(batch).reshape((batch,-1))
        input_tensor = torch.Tensor(self.data[idx])
        res = self.model(input_tensor).mean(0).detach().numpy()[np.newaxis,:]
        return res

    def plot(self, arr):
        fig = plt.figure(figsize=(24, 12), tight_layout=True)
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax3.set_ylim(0, 1)
        ax1.set_title('elegant')
        ax2.set_title('dance')
        ax3.set_title('other')
        ax1.plot(arr[:, 0])
        ax2.plot(arr[:, 1])
        ax3.plot(arr[:, 2])
        fig.savefig(f'flow/{self.filename}.png')

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
    def __init__(self, filename, cam_model, cl=None):
        ansmap = { 'elegant': 0, 'dance': 1 }
        self.filename = filename.replace('.mp4', '')
        cl = cl if cl else ansmap.get(self.filename.split('_')[-1], 2)
        self.targets = [ClassifierOutputTarget(cl) for _ in range(batch)]

        with open(f'out/src/edited/{self.filename}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.cam_model = cam_model

    def dump(self, r=0.4):
        print('grad cam '+ f'{self.filename}.mp4')
        edited = f'out/video/edited/{self.filename}.mp4'
        cam = f'out/video/cam/{self.filename}.mp4'

        cap = cv2.VideoCapture(edited)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(cam, fmt, fps, (size*2, size), 0)

        imgs = np.zeros((frame_count, size, size))
        for fr in tqdm(range(frame_count)):
            if frame_count-arr_size < fr+1:
                mean = frame_count-fr
            else:
                if fr+1 < arr_size:
                    mean = fr+1
                else:
                    mean = arr_size
                imgs[fr:fr+arr_size] += self.run(fr)
            img = np.where(imgs[fr]/mean < r, 0, imgs[fr]*255/mean)
            cat = np.append(img, self.data[fr][0], axis=1)
            writer.write(cat.astype(np.uint8))

    def run(self, fr):
        idx = np.arange(fr,fr+arr_size).repeat(batch).reshape((batch,-1))
        input_tensor = torch.Tensor(self.data[idx])
        res = self.cam_model(input_tensor, self.targets)
        res = torch.tanh(torch.from_numpy(res)).numpy()
        return res