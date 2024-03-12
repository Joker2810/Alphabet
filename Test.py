import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import draw
import matplotlib.pyplot as plt
# Ask the user if they want to train the model or use it
user_input = input("Do you want to train the model or use it? Enter 'train' or 'use': ")


class PrintSampleCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds_train):
        super().__init__()
        self.ds_train = ds_train

    def on_epoch_begin(self, epoch, logs=None):
        # Get one batch of the training data
        x_sample, y_sample = next(iter(self.ds_train))
        # Print the first sample of this batch
        print(f"Sample input shape: {x_sample[0].shape}")
        print(f"Sample input data: {x_sample[0]}")
        print(f"Sample target data: {y_sample[0]}")

        # Display the first image of this batch
        plt.imshow(x_sample[0].numpy().squeeze(), cmap='gray')
        plt.show()


if user_input.lower() == 'train':
    model = load_model('omniglot_model.keras')


    def preprocess(images, labels):
        images = tf.image.rgb_to_grayscale(images)
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels


    # Load the dataset
    ds_train, ds_test = tfds.load(
        'omniglot',
        split=['train', 'test'],
        as_supervised=True
    )

    # Preprocess and shuffle the training dataset
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(10000)

    # Split the dataset into training and validation
    # For example, use 10% of the data for validation
    num_train_examples = ds_train.cardinality().numpy()
    num_val_samples = int(num_train_examples * 0.1)
    ds_val = ds_train.take(num_val_samples)
    ds_train = ds_train.skip(num_val_samples)

    # Batch and prefetch
    ds_train = ds_train.batch(64).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(64).prefetch(tf.data.AUTOTUNE)

    # Preprocess the test dataset
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(64).prefetch(tf.data.AUTOTUNE)

    # Instantiate the custom callback
    print_sample_callback = PrintSampleCallback(ds_train)

    # Add the callback to the fit function
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='omniglot_model.keras', save_best_only=True)
    ]

    # Train the model
    model.fit(ds_train, epochs=1000, validation_data=ds_test, callbacks=callbacks)

    # Evaluate the model
    score = model.evaluate(ds_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    model.save('omniglot_model.keras')
    use_check = input("Do you want to use the model? Enter 'yes' or 'no': ")
    if use_check.lower() == 'yes':
        while True:
            draw.main()
            # Load the image
            img = image.load_img("CenteredDrawinggray.jpg", target_size=(105, 105), color_mode="grayscale")
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img /= 255

            pred = model.predict(img)
            class_index = np.argmax(pred)

            # Add 1 to align with the EMNIST Letters dataset labels and convert to a character
            predicted_letter = chr(class_index + ord('A'))
            predicted_letter = chr(ord(predicted_letter) - 1)

            print(f"The predicted letter is: {predicted_letter}")
            correct = input("Was the prediction correct? Enter 'yes' or 'no': ")
            if correct.lower() == 'no':
                correct_letter = input("What was the correct letter? ")
                print(f"The correct letter was: {correct_letter}")

                # Convert the correct letter to a one-hot encoded vector
                correct_label = to_categorical(ord(correct_letter.upper()) - ord('A'), num_classes=1623)

                # Reshape the image to match the input shape of the model
                img = img.reshape((1, 105, 105, 1))

                # Update the model
                model.train_on_batch(img, np.array([correct_label]))
            user_input = input("Do you want to draw another letter? Enter 'yes' or 'no': ")
            if user_input.lower() == 'no':
                break

elif user_input.lower() == 'use':
    # Load the model
    model = load_model('omniglot_model.keras')
    while True:
        draw.main()
        # Load the image
        img = image.load_img("CenteredDrawinggray.jpg", target_size=(105, 105), color_mode="grayscale")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255

        pred = model.predict(img)
        class_index = np.argmax(pred)

        # Add 1 to align with the EMNIST Letters dataset labels and convert to a character
        predicted_letter = chr(class_index + ord('A'))
        predicted_letter = chr(ord(predicted_letter) - 1)

        print(f"The predicted letter is: {predicted_letter}")
        correct = input("Was the prediction correct? Enter 'yes' or 'no': ")
        if correct.lower() == 'no':
            correct_letter = input("What was the correct letter? ")
            print(f"The correct letter was: {correct_letter}")

            # Convert the correct letter to a one-hot encoded vector
            correct_label = to_categorical(ord(correct_letter.upper()) - ord('A'), num_classes=1623)

            # Reshape the image to match the input shape of the model
            img = img.reshape((1, 105, 105, 1))

            # Update the model
            model.train_on_batch(img, np.array([correct_label]))
        user_input = input("Do you want to draw another letter? Enter 'yes' or 'no': ")
        if user_input.lower() == 'no':
            break

elif user_input.lower() == "display":
    ds_train, ds_test = tfds.load(
        'omniglot',
        split=['train', 'test'],
        as_supervised=True
    )

    # Convert the tf.data.Dataset to numpy arrays
    x_train, y_train = [], []
    for example, label in tfds.as_numpy(ds_train):
        x_train.append(example)
        y_train.append(label)

    # Convert lists to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Reshape and normalize the data
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_train = x_train.astype('float32') / 255

    # Select one sample from each class
    samples = []
    for i in range(1, 27):
        for x in range(2):
            samples.append(x_train[y_train == i][x])

    # Plot the selected samples
    fig, axes = plt.subplots(5, 6, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(26):
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(chr(i + ord('A')))

    plt.tight_layout()
    plt.show()
