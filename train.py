from gan_model import *
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets
from parser import parse



import numpy as np
model2 = GANModel('cross_entropy', 'Adam', 'cross_entropy', 'Adam')
image = model2.generate_image()
plt.imshow(image[0,:,:,:])
plt.show()
num_examples_to_generate = 36
noise_dim = 100
seed = tf.random.normal([num_examples_to_generate, noise_dim])
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 32

train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


model2.train(train_images, 100, BATCH_SIZE, 'True', seed)