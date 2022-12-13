import cv2, os, json, pickle, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from rembg.bg import remove
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .common import size, batch, arr_size, graph
from .read import initial_rang

class Dist:
    def __init__(self, filename, model):
        self.filename = filename.replace('.mp4', '')
        os.mkdir(f'flow/dist/{self.filename}')
        with open(f'out/src/edited/{self.filename}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.frame, self.model = len(self.data), model

    def read(self):
        print('analyze dist of '+ f'{self.filename}.mp4')
        grang, arr, n = 500, np.array([]), 0
        for fr in tqdm(range(self.frame-arr_size)):
            res = self.forward(fr)
            arr = res if arr.size == 0 else np.append(arr, res, axis=0)
        for i in range(len(arr)//grang+1):
            d = arr[i*grang:(i+1)*grang]
            n += len(d)
            self.plot(d, n)

    def forward(self, fr):
        idx = np.arange(fr,fr+arr_size).repeat(batch).reshape((batch,-1))
        input_tensor = torch.Tensor(self.data[idx])
        res = self.model(input_tensor).mean(0).detach().numpy()[np.newaxis,:]
        return res

    def plot(self, arr, n):
        fig = plt.figure(figsize=(24, 12), tight_layout=True)
        graph(fig, arr, ['elegant', 'dance', 'other'], 1.1, -0.1)
        plt.close(fig)
        fig.savefig(f'flow/dist/{self.filename}/{self.filename}_{n}.png')

class Json:
    def __init__(self, filename):
        self.filename = filename.replace('.mp4', '')
        os.mkdir(f'flow/json/{self.filename}')
        self.target = ['RWrist_x', 'LWrist_x', 'RWrist_y', 'LWrist_y']

    def dist(self):
        pos =['Nose_x','Nose_y','P0','Neck_x','Neck_y','P1','RShoulder_x','RShoulder_y',
                'P2','RElbow_x','RElbow_y','P3','RWrist_x','RWrist_y','P4','LShoulder_x',
                'LShoulder_y','P5','LElbow_x','LElbow_y','P6','LWrist_x','LWrist_y','P7',
                'MidHip_x','MidHip_y','P8','RHip_x','RHip_y','P9','RKnee_x','RKnee_y',
                'P10','RAnkle_x','RAnkle_y','P11','LHip_x','LHip_y','P12','LKnee_x',
                'LKnee_y','P13','LAnkle_x','LAnkle_y','P14','REye_x','REye_y','P15',
                'LEye_x','LEye_y','P16','REar_x','REar_y','P17','LEar_x','LEar_y','P18',
                'LBigToe_x','LBigToe_y','P19','LSmallToe_x','LSmallToe_y','P20','LHeel_x',
                'LHeel_y','P21','RBigToe_x','RBigToe_y','P22','RSmallToe_x','RSmallToe_y',
                'P23','RHeel_x','RHeel_y','P24']

        print('analyze json of '+ f'{self.filename}.mp4')
        arr = np.array([])
        idx = list(map(lambda s: pos.index(s), self.target))

        for jname in tqdm(os.listdir(f'json/{self.filename}')):
            with open(f'json/{self.filename}/{jname}') as f:
                data = json.load(f)['people']
                rdata = data[0]['pose_keypoints_2d'][pos.index('RShoulder_x')]
                ldata = data[0]['pose_keypoints_2d'][pos.index('LShoulder_x')]
                res = data[0]['pose_keypoints_2d']

                for n, d in enumerate(data[1:]):
                    rd = d['pose_keypoints_2d'][pos.index('RShoulder_x')]
                    ld = d['pose_keypoints_2d'][pos.index('LShoulder_x')]
                    if ld-rd > ldata-rdata:
                        res = data[n+1]['pose_keypoints_2d']

            d = np.array([])
            for i in idx:
                d = np.append(d, res[i])
            d = d[np.newaxis,:]
            arr = d if arr.size == 0 else np.append(arr, d, axis=0)

        grang, n = 500, 0
        for i in range(len(arr)//grang+1):
            d = arr[i*grang:(i+1)*grang]
            n += len(d)
            self.plot(d, n)

    def cut(self, irang):
        print('analyze part of '+ f'{self.filename}.mp4')
        try:
            rang = irang[self.filename]
        except:
            rang = [int(input('start frame: ')), int(input('end frame: '))]

        video = f'test/{self.filename}.mp4'
        edited = f'flow/video/{self.filename}.mp4'

        cap = cv2.VideoCapture(video)
        f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(edited, fmt, fps, (w, h))

        for i in tqdm(range(rang[1] if rang[1] < f else f)):
            _, frame = cap.read()
            if i > rang[0]:
                writer.write(frame)

    def plot(self, arr, n):
        fig = plt.figure(figsize=(36, 12), tight_layout=True)
        graph(fig, arr, self.target, 2000, -100)
        plt.close(fig)
        fig.savefig(f'flow/json/{self.filename}/{n}.png')

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
        return res