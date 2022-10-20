import torch, torchinfo
from pytorch_grad_cam import GradCAM
from src.py.network import NeuralNetwork
from src.py.common import all_read, arr_size, size, batch
from src.py.analyze import Remove, Cam

# show nn size
# print('Input Size:', batch, arr_size, 1, size, size)
# torchinfo.summary(NeuralNetwork(), (batch, arr_size, 1, size, size))

# dump pred data
# all_read('video', True)
# all_read('archive', True)

# dump rembg data
# Remove('video').dump()
# Remove('archive').dump()
# Remove('').compare()

# run cam
model = NeuralNetwork()
model.load_state_dict(torch.load('out/model/model_weights.pth'))
target_layers = model.convL
cam = GradCAM(model, target_layers)

cam = Cam('aito_dance.mp4', cam)
cam.dump()