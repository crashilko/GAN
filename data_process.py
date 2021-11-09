import tensorflow as tf

def get_train_data():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    print("Data: Loaded sucessfully")
    return train_images, train_labels
def prepare_data(train_images):
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images / 255)
    return train_images