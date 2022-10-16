import os, pickle, numpy as np
from .read import Read

ansmap = { 'elegant': [1, 0, 0], 'dance': [0, 1, 0] }
lenA, out, el = len(ansmap)+1, 128, 16
size, arr_size, pool = 64, 30, 2
diff, last = size/(pool**2)/(el**0.5), 2
sec_d, thr_d = int(out/el/2), int(out/el)
sec_size, thr_size = int(diff/last), last
batch, lenE = 10, size**2

def all_read(dirname, force=False):
    r = Read(dirname, size, force)
    if force if force else input('update data [y/n]: ') == 'y':
        data, teachs, div = np.array([]), np.array([]), np.array([])
        other = [0 for _ in range(lenA-1)]+[1]
        for filename in os.listdir(dirname):
            filename = filename.replace('.mp4', '')
            arr = r.read(filename)
            data = arr if data.size == 0 else np.append(data, arr, axis=0)
            filename = filename.split('_')[-1]
            teach = np.array([ansmap.get(filename, other) for _ in range(len(arr)-arr_size)])
            teachs = teach if teachs.size == 0 else np.append(teachs, teach, axis=0)
            div = np.append(div, len(data)-arr_size)
        with open(f'out/model/{dirname}_read.pkl', 'wb') as f:
            print('all dumping '+ f'out/model/{dirname}_read.pkl')
            pickle.dump((data, teachs, div), f)
    with open(f'out/model/{dirname}_read.pkl', 'rb') as f:
        print('all loading '+ f'out/model/{dirname}_read.pkl')
        return pickle.load(f)

def test_read():
    if input('read video data [y/n]: ') == 'y':
        with open('out/model/video_read.pkl', 'rb') as f:
            print('all loading '+'out/model/video_read.pkl')
            return pickle.load(f)
    else:
        return all_read('test')