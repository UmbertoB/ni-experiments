from itertools import chain, combinations
from layers.default_aug import DefaultAugLayers
from layers.salt_and_pepper import RandomSaltAndPepper
from models.resnet import ResNet50Model
from models.xception import XceptionModel


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


APPROACHES = [subset for subset in all_subsets(
        [
            [],
            [RandomSaltAndPepper()],
            [DefaultAugLayers],
            [DefaultAugLayers, RandomSaltAndPepper()],
        ]
)]

INPUT_SHAPE = (72, 72, 3)
KFOLD_N_SPLITS = 10

CONFIGS = [
    {
        "model": ResNet50Model,
        "active": True,
        "approaches": APPROACHES,
    },
    {
        "model": XceptionModel,
        "active": True,
        "approaches": APPROACHES,
    },
]

