import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import requests

from PIL import Image
from io import BytesIO

from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

path = 'pokeye/input/dataset'  # Path to directory which contains classes
classes = os.listdir(path)  # List of all classes
# print(f'Total number of categories: {len(classes)}')

# A dictionary which contains class and number of images in that class
counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))

# Sort our "counts" dictionary and selecting 5 classes with most number of images
imbalanced = sorted(counts.items(), key=lambda x: x[1], reverse=True)

model = Sequential()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(96, 96, 3), kernel_initializer='he_normal'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(imbalanced), activation='softmax'))

# model.summary()

checkpoint = ModelCheckpoint('pokeye/working/best_model.hdf5', verbose=1, monitor='val_accuracy', save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Loading weights from best model
model.load_weights('pokeye/working/best_model.hdf5')

# Saving model
# model.save('pokeye/working/model.hdf5')


def poke_predictor(f):
    img = np.asarray(bytearray(f.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    img = cv.resize(img, (96, 96))
    img = img.reshape(-1, 96, 96, 3) / 255.0
    preds = model.predict(img)
    pred_class = np.argmax(preds)
    pred_pokemon = imbalanced[int(pred_class)][0]
    percents = round(preds[0][pred_class] * 100, 2)
    print(pred_pokemon, percents)

    return f'Это {pred_pokemon} с вероятностью {percents}%'
