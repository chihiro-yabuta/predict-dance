import sys, os, shutil, torch, torchinfo
from pytorch_grad_cam import GradCAM
from src.py.network import NeuralNetwork
from src.py.common import all_read, arr_size, size, batch, thr_d, el
from src.py.analyze import Flow, Remove, Cam

args = sys.argv[1]

if args == 'show':
    print('Input Size:', batch, arr_size, 1, size, size)
    torchinfo.summary(NeuralNetwork(), (batch, arr_size, 1, size, size))

if args == 'dump':
    all_read('video', True)
    all_read('archive', True)

if args == 'flow':
    model = NeuralNetwork()
    model.load_state_dict(torch.load('out/model/model_weights.pth'))
    shutil.rmtree('flow', ignore_errors=True)
    os.mkdir('flow')

    for s in os.listdir('out/video/edited'):
        flow = Flow(s, model)
        flow.read()

if args == 'remove':
    Remove('video').dump()
    Remove('archive').dump()
    Remove('').compare()

if args == 'cam':
    model = NeuralNetwork()
    model.load_state_dict(torch.load('out/model/model_weights.pth'))
    enc = model.encoder.layers

    def reshape_transform(tensor):
        return tensor.transpose(0,1).reshape((arr_size,batch,thr_d,el))
    cam_model = GradCAM(model, enc, reshape_transform=reshape_transform)

    for s in os.listdir('out/video/edited'):
        cam = Cam(s, cam_model)
        cam.dump()