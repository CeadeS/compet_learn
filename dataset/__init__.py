
from .cars import Cars
from .cub import CUBirds
from .sop import SOProducts
from . import utils
from .base import BaseDataset, BaseTorchDataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

    
_type = {
    'cars': Cars,
    'cub': CUBirds,
    'sop': SOProducts,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST
}



def load(name, root, classes, transform = None):
    dclass = _type[name]
    print(name)
    if name in  ['mnist', 'cifar10', 'cifar100']:
        return BaseTorchDataset(dataset=dclass, root=root, classes=classes, transform=transform)
    else: return dclass(root = root, classes = classes, transform = transform)
