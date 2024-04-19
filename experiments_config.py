from models.resnet import ResNet50Model
from models.xception import XceptionModel
from layers.salt_and_pepper import RandomSaltAndPepper
from layers.default_aug import DefaultAugLayers
import keras.layers as layers

INPUT_SHAPE = (72, 72, 3)
KFOLD_N_SPLITS = 10

CONFIGS = [
    {
        "approach_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper()],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug',
        "data_augmentation_layers": [DefaultAugLayers],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            DefaultAugLayers,
            RandomSaltAndPepper(),
        ],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            DefaultAugLayers,
            layers.GaussianNoise(.1),
        ],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": XceptionModel,
        "active": True,
    },
    {
        "approach_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper()],
        "model": XceptionModel,
        "active": True,
    },
    {
        "approach_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": XceptionModel,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug',
        "data_augmentation_layers": [
            DefaultAugLayers,
        ],
        "model": XceptionModel,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            DefaultAugLayers,
            RandomSaltAndPepper(),
        ],
        "model": XceptionModel,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            DefaultAugLayers,
            layers.GaussianNoise(.1),
        ],
        "model": XceptionModel,
        "active": True,
    },
]
