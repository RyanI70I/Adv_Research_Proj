import tensorflow as tf

import scipy
from keras import layers, Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, add, AveragePooling2D, Rescaling
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from keras import models
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, Softmax
from tensorflow.keras.layers import BatchNormalization


import matplotlib.pyplot as plt

#tf.debugging.set_log_device_placement(True)
# class Res(keras.layers.Layer)
# 	def __init__(self, fill)
#     	self.fill = fill
#     	super(Res, self).__init__()
#
# 	def call(self, inn)
#     	identity = inn
#     	inn = Conv2D(filters=self.fill, kernel_size=2, activation=ReLU)(inn)
#     	# inn = BatchNormalization()(inn)
#     	# inn = Conv2D(filters=self.fill, kernel_size=3,activation=ReLU)(inn)
#     	# inn = BatchNormalization()(inn)
#     	inn = add([inn,identity])
#     	# inn = BatchNormalization()
#     	return inn


batch_size = 32
img_height = 256
img_width = 256
data_dir = r"C/Users/Ryan/Desktop/Data"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    	data_dir,
    	labels="inferred",
    	label_mode="int",
    	shuffle=True,
    	validation_split=.2,
    	subset="training",
    	interpolation="gaussian",
    	seed=123,
    	image_size=(img_height, img_width),
    	batch_size=batch_size)
transform_gen = ImageDataGenerator(
	horizontal_flip=True,
	vertical_flip=True,
	validation_split=.2,


)


train_gen = transform_gen.flow_from_directory(
	data_dir,
	seed = 123,
	batch_size=32,
	class_mode="sparse",
	subset="training"

)
val_gen = transform_gen.flow_from_directory(
	data_dir,
	seed = 123,
	batch_size=32,
	class_mode="sparse",

	subset="validation"

)




val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    	data_dir,
    	seed=123,
    	labels="inferred",
    	label_mode="int",
    	shuffle=True,
    	interpolation="gaussian",
    	validation_split=.2,
    	subset=validation,
    	image_size=(img_height, img_width),
    	batch_size=batch_size)




AUTOTUNE = tf.data.AUTOTUNE



# model = tf.keras.Sequential([
# 	tf.keras.layers.Conv2D(input_shape=(512, 512,3), filters=64, kernel_size=3),
# 	Res(10),
# 	Res(10),
# 	Res(10),
# 	tf.keras.layers.Dense(3, activation='softmax')
# ])

# class ModRes(tf.keras.Model)
# 	def __init__(self)
#     	super(ModRes, self).__init__()
#     	self.Cov1 = tf.keras.layers.Conv2D(input_shape=(8,512, 512,3), filters=64, kernel_size=3)
#     	self.Res1 = Res(128)
#     	# self.Res2 = Res(256)
#     	# self.Res3 = Res(386)
#     	# self.Res4 = Res(512)
#     	self.poool = tf.keras.layers.GlobalAvgPool2D
#     	self.Dense = tf.keras.layers.Dense(3, activation=softmax)
#
# 	def call(self,inputs)
#     	x = self.Cov1(inputs)
#     	x = self.Res1(x)
#     	# x = self.Res2(x)
#     	# x = self.Res3(x)
#     	# x = self.Res4(x)
#     	return self.Dense(x)

def ResBlock(input)
	x = input
	x = Conv2D(filters=128, kernel_size=3, activation=ReLU, padding=same)(input)
	x= layers.Activation(activation=relu)(x)
	x = Conv2D(filters=128, kernel_size=3, activation=ReLU, padding=same)(x)
	x= layers.BatchNormalization()(x)
	input = Conv2D(128, (1,1), activation=ReLU, padding=same)(input)
	final = add([x, input])
	final = layers.Activation(activation=relu)(final)
	return final


def ResBlock256(input)
	x = input
	x = Conv2D(filters=256, kernel_size=3, activation=ReLU, padding=same)(input)
	x= layers.Activation(activation=relu)(x)
	x = Conv2D(filters=256, kernel_size=1, activation=ReLU, padding=same)(x)
	x= layers.BatchNormalization()(x)
	x = layers.AveragePooling2D()(x)
	input = Conv2D(256, (1,1), activation=ReLU, padding=same)(input)
	input = AveragePooling2D()(input)
	final = add([x, input])
	final = layers.Activation(activation=relu)(final)
	return final
def ResBlock512(input)
	x = input
	x = Conv2D(filters=512, kernel_size=2, activation=ReLU, padding=same)(input)
	x= layers.Activation(activation=relu)(x)
	x = Conv2D(filters=512, kernel_size=1, activation=ReLU, padding=same)(x)
	x= layers.BatchNormalization()(x)
	x = layers.AveragePooling2D()(x)

	input = Conv2D(512, (1, 1), activation=ReLU, padding=same)(input)
	input = AveragePooling2D()(input)

	final = add([x, input])
	final = layers.Activation(activation=relu)(final)
	return final







inputs = Input(shape=(256,256,3))
# rescale = Rescaling(1.256)
conv1 = layers.Conv2D( filters=64, kernel_size=7)

# x = rescale(inputs)
x = conv1(inputs)
x = layers.Dropout(.2)(x)

x =  layers.Conv2D( filters=64, kernel_size=7)  (x)
x = layers.Dropout(.2)(x)
x = layers.MaxPooling2D()(x)
x = Conv2D(32,(3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(.2)(x)

x = Conv2D(64,(3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = Conv2D(128,(3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = ResBlock256(x)
x = layers.Dropout(.1)(x)
x = ResBlock256(x)
# x = layers.Dropout(.1)(x)
x = ResBlock256(x)
# x = layers.Dropout(.1)(x)





x= layers.GlobalAvgPool2D()(x)
x = layers.Dropout(.35)(x)

x = Dense(1250)(x)
x = layers.Dropout(.75)(x)

output = Dense(3, activation=softmax)(x)

model = Model(inputs=inputs,outputs=output)

model.summary()

filepath = 'my_best_model.epoch{epoch02d}-loss{val_loss.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks2 = [checkpoint]

model.compile(optimizer='adam',

          	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          	metrics=['accuracy'])

epochs = 30
history = model.fit(
	train_gen,
	callbacks=checkpoint,

	steps_per_epoch=190,
	validation_data=val_gen,
	epochs=epochs,
	validation_steps=50
)

model.save(r"C/Users/Ryan/Desktop"))
