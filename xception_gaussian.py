from keras.layers import GaussianNoise
from models.xception import XceptionModel
from experiment import experiment
import multiprocessing

INPUT_SHAPE = (72, 72, 3)


def run():
    execution_name = 'GaussianNoise'

    # In this execution, the Gaussian Noise has a FIXED FACTOR
    data_augmentation_layers = [GaussianNoise(0.1)]

    xception = XceptionModel(input_shape=INPUT_SHAPE, approach_name=execution_name)

    experiment(xception, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")