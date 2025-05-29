from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

def train_model(model, audio_data, image_data, epochs=20, batch_size=32):
    optimizer = Adam(learning_rate=0.0002)
    loss = MeanSquaredError()

    # Added 'mean_absolute_error' to metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_absolute_error'])

    history = model.fit(audio_data, image_data, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Plot training and validation loss and accuracy
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.legend()
    plt.show()

    return history