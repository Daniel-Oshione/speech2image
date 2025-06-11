from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout


def define_model(input_shape=(20, 44, 1)):
    # Encoder - extract features from audio MFCC
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

    # Decoder - generate image from learned representation
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

    # Output layer - 3 channels for RGB image
    output = Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid', name='generated_image')(x)

    model = Model(inputs=audio_input, outputs=output)
    model.summary()

    return model
