import torchinfo
from src.py.network import NeuralNetwork
from src.py.common import all_read, arr_size, size, batch
from src.py.analyze import Remove

# show nn size
# print('Input Size:', batch, arr_size, 1, size, size)
# torchinfo.summary(NeuralNetwork(), (batch, arr_size, 1, size, size))

# dump pred data
# all_read('video', True)
# all_read('archive', True)

# dump rembg data
# Remove('video').dump()
# Remove('archive').dump()
Remove('').compare()