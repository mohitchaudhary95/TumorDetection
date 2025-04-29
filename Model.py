import os
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from tqdm.keras import TqdmCallback  # Use official TqdmCallback from tqdm.keras

# Optional: Enable mixed precision if supported on your GPU
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print("Mixed precision enabled. Policy:", mixed_precision.global_policy())
except Exception as e:
    print("Mixed precision not enabled:", e)

# CONFIGURATION
DATA_DIR = "/content/drive/MyDrive/brainTumorDataPublic_2299-3064"  # Adjust to your Colab path
CVIND_FILE = os.path.join(DATA_DIR, 'cvind.mat')
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (128, 128)  # We'll resize the 512x512 images for faster training

# --- Data Loader using h5py ---
def load_mat_data(file_path):
    # Load one .mat file using h5py (for MATLAB v7.3+ files)
    with h5py.File(file_path, 'r') as f:
        # Data under 'cjdata' may be stored transposed; use .T to adjust
        image = np.array(f['cjdata']['image']).T.astype(np.float32)
        tumor_mask = np.array(f['cjdata']['tumorMask']).T.astype(np.uint8)
    return image, tumor_mask

# --- tf.data Generator ---
def data_generator(file_list):
    # This generator yields preprocessed image and mask pairs
    for file in file_list:
        try:
            image, mask = load_mat_data(file)
            # Resize image and mask from (512, 512) to IMG_SIZE using tf.image.resize
            image = tf.image.resize(image[..., None], IMG_SIZE).numpy()  # shape: (IMG_SIZE[0], IMG_SIZE[1], 1)
            mask = tf.image.resize(mask[..., None], IMG_SIZE, method='nearest').numpy()  # shape: (IMG_SIZE[0], IMG_SIZE[1], 1)
            # Normalize image to [0, 1]
            image = image / 255.0
            # Convert mask to float32 (binary segmentation: 0.0 or 1.0)
            mask = mask.astype(np.float32)
            yield image, mask
        except Exception as e:
            print(f"Error processing {file}: {e}")

# --- Build tf.data Dataset ---
def get_dataset(file_list, batch_size=BATCH_SIZE, shuffle_buffer=100):
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(file_list),
        output_types=(tf.float32, tf.float32),
        output_shapes=((IMG_SIZE[0], IMG_SIZE[1], 1), (IMG_SIZE[0], IMG_SIZE[1], 1))
    )
    ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Simple U-Net Model for 2D Segmentation ---
def build_unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    # Bottleneck
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D()(c3)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Prepare File Lists ---
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.mat') and f != 'cvind.mat']
print(f"Found {len(all_files)} .mat files.")

# We'll perform a random 80/20 train/test split (you can modify this if you wish to use cvind.mat)
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Create tf.data Datasets
train_ds = get_dataset(train_files, batch_size=BATCH_SIZE)
test_ds = get_dataset(test_files, batch_size=BATCH_SIZE)

# --- Build Model ---
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
model = build_unet_model(input_shape)
model.summary()

# --- Callbacks ---
checkpoint = tf.keras.callbacks.ModelCheckpoint('brain_tumor_segmentation_model.h5', monitor='val_accuracy',
                                                    save_best_only=True, mode='max')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tqdm_callback = TqdmCallback()  # No additional keyword arguments

# --- Train the Model ---
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=test_ds,
                    callbacks=[checkpoint, early_stopping, tqdm_callback],
                    verbose=0)  # TqdmCallback provides the progress display

# --- Evaluate the Model ---
loss, accuracy = model.evaluate(test_ds)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# --- Visualize Predictions ---
def visualize_prediction(dataset, model, num_samples=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        for i in range(num_samples):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(images[i, ..., 0], cmap='gray')
            plt.title("Input MRI")
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i, ..., 0], cmap='jet')
            plt.title("Ground Truth Mask")
            plt.subplot(1, 3, 3)
            plt.imshow(preds[i, ..., 0] > 0.5, cmap='jet')
            plt.title("Predicted Mask")
            plt.show()

visualize_prediction(test_ds, model)

# --- Save the Model ---
model.save('brain_tumor_segmentation_model.h5')
print("Model saved as brain_tumor_segmentation_model.h5")