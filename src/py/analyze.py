import cv2, os, pickle, numpy as np, torch
from tqdm import tqdm
from rembg.bg import remove
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .common import size, batch, arr_size, lenA
from .read import initial_rang
from .network import NeuralNetwork

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
    def __init__(self, filename):
        self.filename = filename
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load('out/model/model_weights.pth'))
        self.target_layers = self.model.convL

    def dump(self):
        print(self.run(0).shape)

    def run(self, flame):
        with open(f'out/src/edited/{self.filename}.pkl', 'rb') as f:
            data = pickle.load(f)
        idx = np.array(list(map(lambda e: np.arange(flame, flame+arr_size), range(batch))))
        input_tensor = torch.Tensor(data[idx])
        targets = [ClassifierOutputTarget(lenA-1) for _ in range(batch)]

        cam = GradCAM(self.model, self.target_layers)
        return cam(input_tensor, targets)