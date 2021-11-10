import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def load_config():
    pass

def create_gen_arcitecture():
    pass

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    print("Generator: Built sucessfully")
    return model

def generate_image(generator):

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    print("Generator: Image generated sucessfully")
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()
    return generated_image

def gen_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def set_gen_optimizer(name):
    if name == 'Adam':
        return tf.keras.optimizers.Adam(1e-4)
    else:
        print("Generator: optimizer not implemented")