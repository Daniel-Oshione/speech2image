import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K

def build_generator(input_shape=(20, 44, 1)):
    audio_input = Input(shape=input_shape, name='audio_input')
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(audio_input)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(16*16*64, activation='relu')(x)
    x = Reshape((16, 16, 64))(x)

    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid', name='generated_image')(x)

    model = Model(inputs=audio_input, outputs=output, name='generator')
    return model

def build_discriminator(input_shape=(64, 64, 3)):
    image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=image_input, outputs=x, name='discriminator')
    return model

# Move VGG16 model creation outside the loss function to avoid tf.function variable creation error
vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
vgg_model.trainable = False

def perceptual_loss(y_true, y_pred):
    # Convert float32 images in [0,1] to uint8 in [0,255] for preprocess_input
    y_true_uint8 = tf.image.convert_image_dtype(y_true, dtype=tf.uint8, saturate=True)
    y_pred_uint8 = tf.image.convert_image_dtype(y_pred, dtype=tf.uint8, saturate=True)
    y_true_pp = preprocess_input(tf.cast(y_true_uint8, tf.float32))
    y_pred_pp = preprocess_input(tf.cast(y_pred_uint8, tf.float32))
    true_features = vgg_model(y_true_pp)
    pred_features = vgg_model(y_pred_pp)
    return K.mean(K.square(true_features - pred_features))

class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, lambda_perceptual=0.1):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_perceptual = lambda_perceptual
        self.mse = MeanSquaredError()

    def compile(self, g_optimizer, d_optimizer):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = MeanSquaredError()

    def train_step(self, data):
        audio, real_images = data

        batch_size = tf.shape(audio)[0]

        # Generate fake images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(audio, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            # Ensure labels have the same shape as discriminator outputs
            real_labels = tf.ones_like(real_output)
            fake_labels = tf.zeros_like(fake_output)
            gen_labels = tf.ones_like(fake_output)

            # Fix shape mismatch by slicing labels and outputs to minimum batch size
            min_batch_size = tf.minimum(tf.shape(real_labels)[0], tf.shape(fake_output)[0])
            min_batch_size = tf.minimum(min_batch_size, tf.shape(real_output)[0])
            real_labels = real_labels[:min_batch_size]
            fake_labels = fake_labels[:min_batch_size]
            gen_labels = gen_labels[:min_batch_size]

            real_output = real_output[:min_batch_size]
            fake_output = fake_output[:min_batch_size]

            # Also slice real_images and fake_images to min_batch_size for perceptual loss
            real_images = real_images[:min_batch_size]
            fake_images = fake_images[:min_batch_size]

            # Generator losses
            perceptual = perceptual_loss(real_images, fake_images)
            gen_loss = self.loss_fn(real_images, fake_images) + self.lambda_perceptual * perceptual
            gen_loss += tf.keras.losses.binary_crossentropy(gen_labels, fake_output)

            # Discriminator loss
            disc_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_output)
            disc_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_output)
            disc_loss = disc_loss_real + disc_loss_fake

        # Calculate gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}
