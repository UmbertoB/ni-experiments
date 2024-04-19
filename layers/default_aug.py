import keras.layers as layers
from keras.models import Sequential


DefaultAugLayers = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=0.02),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2)
], name="augmentation_layers")