from ..src.py.common import all_read
from .review import Review

#all_read('test', True)
#Review('test').dump()

Review('video').dump()
Review('archive').dump()
Review('').compare()