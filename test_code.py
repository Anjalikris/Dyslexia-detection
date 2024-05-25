import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

model = load_model('model_cnn_dyslexia_mra.h5')

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(
    directory='test_',
    target_size=(29, 29),
    batch_size=1,
    class_mode=None,  # This should be None since you're not using labels
    shuffle=False,
    seed=123
)

pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
print(pred)
predicted_class_indices = np.argmax(pred, axis=1)
print(predicted_class_indices)
label = ['DYSLEXIC', 'NORMAL']
print(label[predicted_class_indices[0]])
