import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import requests

from pathlib import Path
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
from keras.utils import to_categorical

warnings.filterwarnings('ignore')

path = Path('input/dataset')  # Путь к директории с датасетом
classes = sorted(os.listdir(path))  # Список всех классов
print(f'Всего категорий: {len(classes)}')
# Всего категорий: 149

# Словарь, содержащий название класса и количество изображений
counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))

print(f'Всего изображений в датасете: {sum(list(counts.values()))}')
# Всего изображений в датасете: 10693

# Построение графика количества изображений в каждом классе
fig = plt.figure(figsize=(25, 5))
sns.barplot(x=list(counts.keys()), y=list(counts.values())).set_title('Количество изображений')
plt.xticks(rotation=90)
plt.margins(x=0)
plt.show()

# Сортируем наш словарь по количеству иозбражений и выбираем 5 классов,
# имеющих максимальное число изображений
imbalanced = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
print(imbalanced)
# [('Mewtwo', 307), ('Pikachu', 298), ('Charmander', 296), ('Bulbasaur', 289), ('Squirtle', 280)]

# Сохраним только названия классов
imbalanced = [i[0] for i in imbalanced]
print(imbalanced)
# ['Mewtwo', 'Pikachu', 'Charmander', 'Bulbasaur', 'Squirtle']

X = []  # Список изображений
Y = []  # Список названий классов

# Цикл по классам
for c in classes:
    # Используем только сохраненные в imbalanced классы
    if c in imbalanced:
        dir_path = os.path.join(path, c)
        label = imbalanced.index(c)

        # Читаем, изменяем размер и добавляем изображение и название класса в списки
        for i in os.listdir(dir_path):
            image = cv.imread(os.path.join(dir_path, i))

            try:
                resized = cv.resize(image, (96, 96))  # Сжимаем изображением до (96, 96)
                X.append(resized)
                Y.append(label)

            # Если изображение нельзя прочитать – пропускаем
            except:
                print(os.path.join(dir_path, i), '[ОШИБКА] нельзя прочитать файл')
                continue

print('[ЗАВЕРШЕНО]')

obj = Counter(Y)

# Plotting number of images in each class
# fig = plt.figure(figsize=(15, 5))
# sns.barplot(x=[imbalanced[i] for i in obj.keys()], y=list(obj.values())).set_title('Number of images in each class')
# plt.margins(x=0)
# plt.show()

# Конвертируем список изображений в массив numpy
X = np.array(X).reshape(-1, 96, 96, 3)

# Масштабируем массив
X = X / 255.0

y = to_categorical(Y, num_classes=len(imbalanced))

# Разделяем данные на наборы данных для обучения и теста
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=666)

# Генерируем дополнительные изображения
datagen = ImageDataGenerator(rotation_range=45,  # Случайный угол
                             zoom_range=0.2,  # Случайное масштабирование
                             horizontal_flip=True,  # Случайный наклон по горизонтали
                             width_shift_range=0.15,  # Случайное смещение по ширине
                             height_shift_range=0.15,  # Случайное смещение по высоте
                             shear_range=0.2)  # Интенсивность сдвига

datagen.fit(X_train)

# This piece of code can be used if you eant to look what your datagen doing with your images
# img = X[600]
# img = img.reshape([-1, 96, 96, 3])

# i = 0
# fig = plt.figure(figsize = (18, 8))

# for i, flow in enumerate(datagen.flow(img, batch_size = 1)):
#     fig.add_subplot(2, 5, i+1)
#     plt.imshow(np.squeeze(flow[:, :, ::-1]))
#     plt.axis('off')
#     i += 1
#     if i >= 10:
#         break

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

# Сохраняем лучшую модель
# checkpoint = ModelCheckpoint(Path('working/best_model.hdf5'), verbose=1, monitor='val_accuracy', save_best_only=True)

# В качестве функции потерь используем категориальную перекрестную энтропию
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=100,
#                               validation_data=(X_test, y_test),
#                               steps_per_epoch=len(X_train) // 32, callbacks=[checkpoint])

# Plot learning curves
# fig = plt.figure(figsize=(17, 4))
#
# plt.subplot(121)
# plt.plot(history.history['accuracy'], label='acc')
# plt.plot(history.history['val_accuracy'], label='val_acc')
# plt.legend()
# plt.grid()
# plt.title(f'accuracy')
#
# plt.subplot(122)
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.grid()
# plt.title(f'loss')

# Loading weights from best model
model.load_weights(Path('working/best_model.hdf5'))

# Сохранение модели
model.save(Path('working/model.hdf5'))




# Датафрейм изображений
mewtwo = ['https://files.cults3d.com/uploaders/17560495/illustration-file/8e1f6705-9f7e-4721-999a-35d1043ab419/meewtwo.jpg',
          'https://images.immediate.co.uk/production/volatile/sites/3/2022/06/Pokemon-Go-Mewtwo-counters-af97a91.jpg',
          'https://cdn.vox-cdn.com/thumbor/W-3KoS_ImZ6ZsL4zujBCknYCpcg=/0x0:1750x941/1400x1400/filters:focal(735x331:1015x611):format(png)/cdn.vox-cdn.com/uploads/chorus_image/image/53111665/Mewtwo_M01.0.0.png']

pikachu = ['https://lh3.googleusercontent.com/proxy/DrjDlKlu9YonKbj3iNCJNJ3DGqzy9GjeXXSUv-TcVV4UN9PMCAM5yIkGLPG7wYo3UeA4sq5OmUWM8M6K5hy2KOAhf8SOL3zPH3axb2Xo3HX2XTU8M2xW4X6lVg=w720-h405-rw',
           'https://trashbox.ru/files/626440_8c7af7/pnwrpse.png',
           'https://johnlewis.scene7.com/is/image/JohnLewis/237525467']

charmander = ['https://img.pokemondb.net/artwork/large/charmander.jpg',
              'https://www.pokemoncenter.com/wcsstore/PokemonCatalogAssetStore/images/catalog/products/P5073/701-03990/P5073_701-03990_01.jpg',
              'https://static.posters.cz/image/750/%D0%A7%D0%B0%D1%88%D0%BA%D0%B0/pokemon-charmander-glow-i72513.jpg']

bulbasaur = ['https://img.pokemondb.net/artwork/large/bulbasaur.jpg',
             'https://ae01.alicdn.com/kf/HTB1aWullxSYBuNjSsphq6zGvVXaR/Big-Size-55-CM-Plush-Toy-Squirtle-Bulbasaur-Charmander-Toy-Sleeping-Pillow-Doll-For-Kid-Birthday.jpg',
             'https://archives.bulbagarden.net/media/upload/f/f7/Bulbasaur_Detective_Pikachu.jpg']

squirtle = ['https://assets.pokemon.com/assets/cms2/img/pokedex/full/007.png',
            'https://cdn.vox-cdn.com/thumbor/l4cKX7ZWargjs-zlxOSW2WZVgfI=/0x0:2040x1360/1200x800/filters:focal(857x517:1183x843)/cdn.vox-cdn.com/uploads/chorus_image/image/61498573/jbareham_180925_ply0802_0030.1537570476.jpg',
            'https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fdavidthier%2Ffiles%2F2018%2F07%2FSquirtle_Squad.jpg']

test_df = [mewtwo, pikachu, charmander, bulbasaur, squirtle]

# Списки для хранения данных
val_x = []
val_y = []

for i, urls in enumerate(test_df):
    for url in urls:
        r = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(r.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        val_x.append(image)
        val_y.append(i)

rows = 5
cols = 3

fig = plt.figure(figsize=(25, 25))

for i, j in enumerate(zip(val_x, val_y)):  #
    orig = j[0]  # Оригинал изображения
    label = j[1]  # Название класса

    image = cv.resize(orig, (96, 96))  # Сжимаем до (96, 96)
    image = image.reshape(-1, 96, 96, 3) / 255.0  # Изменяем форму и масштаб
    preds = model.predict(image)  # Изображение для распознавания
    pred_class = np.argmax(preds)  # Определение класса

    true_label = f'Истинный класс: {imbalanced[label]}'
    pred_label = f'Предсказанный: {imbalanced[pred_class]} {round(preds[0][pred_class] * 100, 2)}%'

    fig.add_subplot(rows, cols, i + 1)
    plt.imshow(orig[:, :, ::-1])
    plt.title(f'{true_label}\n{pred_label}', fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.show()
