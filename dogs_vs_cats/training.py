# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

import os
from datetime import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np

LEARNING_DATASET_DIR_PATH = '/home/youngkiu/dataset/dataset_dogs_vs_cats/'
INPUT_SHAPE = (200, 200)
BATCH_SIZE = 64

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

train_dir = os.path.join(LEARNING_DATASET_DIR_PATH, 'train')
valid_dir = os.path.join(LEARNING_DATASET_DIR_PATH, 'test')

print(len(os.listdir(os.path.join(train_dir, 'cats'))))
print(len(os.listdir(os.path.join(train_dir, 'dogs'))))
print(len(os.listdir(os.path.join(valid_dir, 'cats'))))
print(len(os.listdir(os.path.join(valid_dir, 'dogs'))))


train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    target_size=INPUT_SHAPE,
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    target_size=INPUT_SHAPE,
)


def make_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                            padding='same', input_shape=INPUT_SHAPE+(3,)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(128, (3, 3), activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu',
                           kernel_initializer='he_uniform'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(lr=1e-3, momentum=0.9),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


model = make_model()
model.summary()


logdir = 'logs/{}/'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=logdir+'{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

csv_logger = keras.callbacks.CSVLogger(logdir + 'training.log')

history = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch=len(train_generator),
    epochs=1000,
    validation_steps=len(valid_generator),
    callbacks=[checkpoint_callback, tensorboard_callback, csv_logger],
    verbose=2,
    use_multiprocessing=False
)


loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(len(precision))

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()
