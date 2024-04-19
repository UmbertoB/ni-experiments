import tensorflow as tf
import random
from keras_cv.layers import BaseImageAugmentationLayer
from keras_cv.core import UniformFactorSampler
import numpy as np


class RandomSaltAndPepper(BaseImageAugmentationLayer):

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs, seed=seed)
        self.seed = seed
        self.factor = UniformFactorSampler(0, .5)

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_image(self, image, transformation=None, **kwargs):
        mask = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)
        noisy_outputs = tf.where(mask < random.random() * self.factor() / 2, 0.0, image)
        noisy_outputs = tf.where(mask > 1 - random.random() * self.factor() / 2, 1.0, noisy_outputs)
        return noisy_outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'noise_level': self.noise_level,
            'seed': self.seed,
        }
        base_config = super(RandomSaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
