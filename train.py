from gan_model import *
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets
from parser import parse



model = GANModel('cross_entropy', 'Adam', 'cross_entropy', 'Adam')
image = model.generate_image()
plt.imshow(image[0,:,:,0])
plt.show()
num_examples_to_generate = 9
noise_dim = 100
seed = tf.random.normal([num_examples_to_generate, noise_dim])

(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

model.train(train_dataset, 25, 32, 'True', seed)