import numpy as np
import cv2 as cv
import os
import warnings

from pathlib import Path

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')

path = Path('input/dataset')  # Path to directory which contains classes
classes = sorted(os.listdir(path))  # List of all classes
# print(f'Total number of categories: {len(classes)}')

# A dictionary which contains class and number of images in that class
counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))


# Построение графика количества изображений в каждом классе
# fig = plt.figure(figsize=(25, 5))
# # sns.lineplot(x=list(counts.keys()), y=list(counts.values())).set_title('Количество изображений')
# sns.barplot(x=list(counts.keys()), y=list(counts.values())).set_title('Количество изображений')
# plt.xticks(rotation=90)
# plt.margins(x=0)
# plt.show()

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

# checkpoint = ModelCheckpoint(Path('pokeye/working/best_model.hdf5'), verbose=1, monitor='val_accuracy', save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Loading weights from best model
model.load_weights(Path('pokeye/working/best_model.hdf5'))

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

    return f'It\'s {pred_pokemon}! I\'m {percents}% certain of this'
