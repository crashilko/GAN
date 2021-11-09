import tensorflow as tf
def get_train_data:
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels

