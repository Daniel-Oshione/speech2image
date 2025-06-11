import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def custom_gan_training_loop(gan, dataset, epochs=20, batch_size=32, checkpoint_dir='checkpoints'):
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    gan.compile(g_optimizer, d_optimizer)

    train_loss = []
    disc_loss = []

    # Setup checkpoint manager
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=gan.generator,
                                     discriminator=gan.discriminator,
                                     g_optimizer=g_optimizer,
                                     d_optimizer=d_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Restore latest checkpoint if exists
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Starting training from scratch.")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_g_loss = 0
        epoch_d_loss = 0
        steps = 0

        for batch, (audio_batch, image_batch) in enumerate(dataset.batch(batch_size)):
            losses = gan.train_step((audio_batch, image_batch))
            epoch_g_loss += losses["gen_loss"].numpy()
            epoch_d_loss += losses["disc_loss"].numpy()
            steps += 1

            if batch % 10 == 0:
                gen_loss_val = losses['gen_loss']
                disc_loss_val = losses['disc_loss']
                if hasattr(gen_loss_val, 'numpy'):
                    gen_loss_val = gen_loss_val.numpy()
                if hasattr(disc_loss_val, 'numpy'):
                    disc_loss_val = disc_loss_val.numpy()
                # Defensive conversion to float for printing
                def to_scalar(x):
                    if hasattr(x, 'numpy'):
                        x = x.numpy()
                    if hasattr(x, 'shape') and x.shape != ():
                        try:
                            x = x.item()
                        except Exception:
                            x = x.mean()
                    if isinstance(x, (list, tuple, np.ndarray)):
                        x = float(np.mean(x))
                    return float(x)
                gen_loss_float = to_scalar(gen_loss_val)
                disc_loss_float = to_scalar(disc_loss_val)
                print(f"Batch {batch}, Gen Loss: {gen_loss_float:.4f}, Disc Loss: {disc_loss_float:.4f}")

        avg_g_loss = epoch_g_loss / steps
        avg_d_loss = epoch_d_loss / steps
        train_loss.append(avg_g_loss)
        disc_loss.append(avg_d_loss)

        # Defensive conversion to float for printing epoch losses
        def to_scalar(x):
            if hasattr(x, 'numpy'):
                x = x.numpy()
            if hasattr(x, 'shape') and x.shape != ():
                try:
                    x = x.item()
                except Exception:
                    x = x.mean()
            if isinstance(x, (list, tuple, np.ndarray)):
                x = float(np.mean(x))
            return float(x)
        avg_g_loss_val = to_scalar(avg_g_loss)
        avg_d_loss_val = to_scalar(avg_d_loss)
        print(f"Epoch {epoch+1} Gen Loss: {avg_g_loss_val:.4f}, Disc Loss: {avg_d_loss_val:.4f}")

        # Save checkpoint every epoch
        save_path = manager.save()
        print(f"Saved checkpoint for epoch {epoch+1}: {save_path}")

    # Plot losses
    plt.plot(train_loss, label='Generator Loss')
    plt.plot(disc_loss, label='Discriminator Loss')
    plt.legend()
    plt.show()

    return train_loss, disc_loss
