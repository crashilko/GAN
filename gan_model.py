import tensorflow as tf
from tensorflow.keras import layers
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    device_name = '/device:CPU:0'

class GANModel:
    def __init__(self, gen_loss, gen_opt, discr_loss, discr_opt):
        self.generator, self.gen_loss, self.gen_opt = self.__init_g__(gen_loss, gen_opt)
        self.discriminator, self.discr_loss, self.discr_opt = self.__init_d__(discr_loss, discr_opt)
        self.checkpoint, self.checkpoint_prefix = self.save_checkpoint()
        print("Init sucess")
        print("Model summary\nGenerator")
        print(self.generator.summary())
        print("Discriminator")
        print(self.discriminator.summary())

    def __init_g__(self, generator_loss, generator_optimizer):
        loss = self.__set_loss__(generator_loss)
        opt = self.__set_opt__(generator_optimizer)
        model = self.__set_architecture_g__()
        return model, loss, opt

    def __init_d__(self, discriminator_loss, discriminator_optimizer):
        loss = self.__set_loss__(discriminator_loss)
        opt = self.__set_opt__(discriminator_optimizer)
        model = self.__set_architecture_d__()
        return model, loss, opt

    def __set_opt__(self, opt_name):
        if opt_name == 'Adam':
            return tf.keras.optimizers.Adam(1e-4)
        else:
            print('Not impelmented optimizer. Init Failed')
            return -1

    def __set_loss__(self, loss_name):
        if loss_name == "cross_entropy":
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            print("Not implemented loss type. Init Failed")
            return -1

    def generator_loss(self, fake_output):
        return self.gen_loss(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.discr_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.discr_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def __set_architecture_g__(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8 * 8 * 128, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 128)))
        #assert model.output_shape == (None, 7, 7, 64)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        print(model.output_shape)
        #assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        print(model.output_shape)
        #assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())


        model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        print(model.output_shape)
        assert model.output_shape == (None, 32, 32, 3)
        return model

    def __set_architecture_d__(self):
        model = tf.keras.Sequential()
        # normal
        model.add(layers.Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(256, (3,3), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # classifier
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def compile(self, BATCH_SIZE):
        pass

    def generate_image(self):
        noise = tf.random.normal([1, 100])
        generated_image = self.generator(noise, training=False)
        return generated_image

    def save_checkpoint(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_opt,
                                         discriminator_optimizer=self.discr_opt,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        return checkpoint, checkpoint_prefix


    @tf.function
    def train_step(self, images, BATCH_SIZE, noise_dim):
        with tf.device(device_name):
          noise = tf.random.normal([BATCH_SIZE, noise_dim])

          with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
              generated_images = self.generator(noise, training=True)

              real_output = self.discriminator(images, training=True)
              fake_output = self.discriminator(generated_images, training=True)

              gen_loss = self.generator_loss(fake_output)
              disc_loss = self.discriminator_loss(real_output, fake_output)
              #print("generator loss:",gen_loss, "\ndiscriminator loss:", disc_loss)

          gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
          gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

          self.gen_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
          self.discr_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs, BATCH_SIZE, SHOW_IMAGES, SEED = None):
        with tf.device(device_name):
          for epoch in range(epochs):
              print("epoch", epoch)
              start = time.time()

              for image_batch in tqdm(dataset):
                  self.train_step(image_batch, BATCH_SIZE, 100)
              # Produce images for the GIF as you go
              # display.clear_output(wait=True)
              if SHOW_IMAGES:
                  self.generate_and_save_images(epoch + 1, SEED)

              # Save the model every 15 epochs
              if (epoch + 1) % 15 == 0:
                  self.checkpoint.save(file_prefix=self.checkpoint_prefix)

              print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

          # Generate after the final epoch
          # display.clear_output(wait=True)
          if SHOW_IMAGES:
              self.generate_and_save_images(epoch, SEED)


    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(6, 6, i + 1)

            plt.imshow(tf.cast(predictions[i, :, :, :]*127.5 + 127.5, dtype=tf.int32))
            plt.axis('off')

        plt.savefig('generated_images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()