from keras.callbacks import EarlyStopping
import multiprocessing
from models.xception import XceptionModel
from dataset.cifar import get_cifar10, get_cifar10_corrupted
from layers.salt_and_pepper import RandomSaltAndPepper
import keras_cv.layers as layers
import tensorflow as tf
from utils.configs import set_memory_growth
from utils.metrics import write_acc_avg, write_acc_each_dataset, write_acc_each_dataset_line
from utils.consts import CORRUPTIONS_TYPES
from keras_cv.core import NormalFactorSampler, UniformFactorSampler


BATCH_SIZE = 128
IMG_SIZE = 72
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def run():
    set_memory_growth(tf)

    # @TODO: TESTAR COM UNIFORM FACTOR SAMPLER
    # uniform_factor = UniformFactorSampler(0, .9)
    factor = NormalFactorSampler(mean=0.3, stddev=0.1, min_value=.0, max_value=.9)

    execution_name = 'DefaultAug+S&P'

    data_augmentation_layers = [
        RandomSaltAndPepper(factor),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ]

    train_ds, val_ds = get_cifar10(BATCH_SIZE, data_augmentation_layers)

    xception = XceptionModel(input_shape=INPUT_SHAPE)

    xception.compile()

    xception.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=100,
        callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
    )

    for corruption in CORRUPTIONS_TYPES:
        xception.evaluate(
            get_cifar10_corrupted(BATCH_SIZE, corruption),
            f'cifar10/{corruption}',
            data_augmentation_layers,
            execution_name,
        )

    write_acc_avg()
    write_acc_each_dataset()
    write_acc_each_dataset_line()


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
