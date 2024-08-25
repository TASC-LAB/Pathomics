import keras
import tensorflow as tf

from .generator     import Generator
from .discriminator import Discriminator

class StainToStainGAN(keras.models.Model):
    def __init__(self, image_shape):
        super(StainToStainGAN, self).__init__()
        self.image_shape = image_shape

        # Create Discriminator and Generator instances
        self.generator     = Generator(image_shape)
        self.discriminator = Discriminator(image_shape)

        # Make discriminator weights not trainable
        self.discriminator.model.trainable = False

        # Define source image input
        in_src = tf.keras.layers.Input(shape=image_shape)

        # Connect source image to generator input
        gen_out = self.generator.model(in_src)

        # Connect source input and generator output to discriminator input
        dis_out = self.discriminator.model([in_src, gen_out])

        # Create combined GAN model
        self.gan_model = keras.models.Model(in_src, [dis_out, gen_out])

        # Compile model
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        self.gan_model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])

    def call(self, real_images):
        # Generate fake images.
        noise = tf.random.normal(shape=(tf.shape(real_images)[0], *self.generator.input_shape[1:]))
        fake_images = self.generator.model(noise)

        # Train discriminator
        d_loss_real = self.discriminator.model.train_on_batch(real_images, tf.ones((tf.shape(real_images)[0],  1)))
        d_loss_fake = self.discriminator.model.train_on_batch(fake_images, tf.zeros((tf.shape(real_images)[0], 1)))
        d_loss      = 0.5 * tf.add(d_loss_real, d_loss_fake)

        # Train generator (via GAN model)
        g_loss = self.gan_model.train_on_batch(noise, [tf.ones((tf.shape(real_images)[0], 1)), real_images])

        return d_loss, g_loss

# Example usage:
# image_shape = (256, 256, 3)  # Example image shape
# gan = StainToStainGAN(image_shape)

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = tf.keras.layers.Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = keras.models.Model(in_src, [dis_out, gen_out])
    # compile model
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model