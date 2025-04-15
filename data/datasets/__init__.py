# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
# from .cuhk03 import CUHK03
from .cuhk03_np_labeled import CUHK03NpLabeled
from .cuhk03_np_detected import CUHK03NpDetected
from .msmt17 import MSMT17
from .veri import VeRi
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03NpLabeled': CUHK03NpLabeled,
    'cuhk03NpDetected': CUHK03NpDetected,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
