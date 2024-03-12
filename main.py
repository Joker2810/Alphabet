import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model

# Enable XLA and Mixed Precision
tf.config.optimizer.set_jit(True)


# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs {gpus} will be used")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
else:
    print("No GPUs found. Using CPU instead.")

# Load and preprocess the dataset
AUTO = tf.data.experimental.AUTOTUNE
def preprocess(images, labels):
    images = tf.image.rgb_to_grayscale(images)
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

try:
    ds_train, ds_test = tfds.load('omniglot', split=['train', 'test'], as_supervised=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=AUTO).cache().batch(32).prefetch(AUTO)
    ds_test = ds_test.map(preprocess, num_parallel_calls=AUTO).cache().batch(32).prefetch(AUTO)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

def build_model():
    inputs = Input(shape=(105, 105, 1))  # Variable input size
    # Convolutional Block 1
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Convolutional Block 2
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Convolutional Block 3
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Convolutional Block 4
    x = Conv2D(1028, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Convolutional Block 5
    x = Conv2D(2056, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.5)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(2056, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2056, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1028, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1028, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1623, activation='softmax')(x)  # Output layer for 1 class
    model = Model(inputs=inputs, outputs=outputs)
    return model


if os.path.isfile('omniglot_model.keras'):
    model = load_model('omniglot_model.keras')
else:
    model = build_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='omniglot_model.keras', save_best_only=True)
]

# Train the model
model.fit(ds_train, epochs=1000, validation_data=ds_test, callbacks=callbacks)

# Evaluate the model
model = load_model('omniglot_model.keras')
score = model.evaluate(ds_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])