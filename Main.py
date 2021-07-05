import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import glob
import os
import shutil
from tensorflow.keras import layers

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

batch_size = 32
img_height = 512
img_width = 512
data_dir = "/Users/ryankersten/Desktop/Data/"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 3

    model = tf.keras.applications.resnet.ResNet50(
        include_top=True, weights=None, input_tensor=None,
        input_shape=(512, 512, 3), pooling=None, classes=3
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print(model.summary())

    epochs = 10
    history = model.fit(
        train_ds,
        epochs=epochs
    )
