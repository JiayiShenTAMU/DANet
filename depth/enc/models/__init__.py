from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .danet import *
#from .fcn_with_fuse import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    from .fcn_with_fuse import get_fcn_with_fuse
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'danet': get_danet,
        'fcn_with_fuse': get_fcn_with_fuse
    }
    return models[name.lower()](**kwargs)
