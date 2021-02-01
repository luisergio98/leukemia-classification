import numpy as np
from numpy import expand_dims
import cv2
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image as kerasImage
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Image:
    img = None
    name = ''
    type = ''

    def __init__(self, name, img, type):
        self.name = name
        self.img = img
        self.type = type


def load(folder):
    print("Loading images...")
    pictures = []
    for filename in enumerate(os.listdir(folder)):
        pictures.append(Image(filename[1], cv2.imread(os.path.join(folder, filename[1])), filename[1][-5]))
    return pictures


def process_and_save(folder, pictures):
    print("Processing and saving images...")
    for index, image in enumerate(pictures):
        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(img, (7, 7), 0)
        (T, binI) = cv2.threshold(suave, 105, 255, cv2.THRESH_BINARY_INV)
        subfolder = folder + 'Leukemia/' if image.type == '1' else folder + 'Normal/'
        cv2.imwrite(subfolder + image.name,
                    cv2.bitwise_and(img, img, mask=binI))


def generate_augmented_images(folder, pictures):
    print("Generate augmented images...")
    datagen = ImageDataGenerator(rotation_range=30, horizontal_flip=True, brightness_range=[0.5, 1.0],
                                 zoom_range=[0.5, 1.0])
    for index, image in enumerate(pictures):
        samples = expand_dims(img_to_array(image.img), 0)
        augmented_image = datagen.flow(samples, batch_size=1).next()[0].astype('uint8')
        cv2.imwrite(folder + image.name, augmented_image)

        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(img, (7, 7), 0)
        (T, binI) = cv2.threshold(suave, 105, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('Modified' + folder + image.name,
                    cv2.bitwise_and(img, img, mask=binI))


def prepare_data(images_folder, modified_images_folder, augmented_images_folder):
    print('Preparing data...')
    images = load(images_folder)
    process_and_save(modified_images_folder, images)
    generate_augmented_images(augmented_images_folder, images)
    print('Data prepared!')


if __name__ == '__main__':

    images_folder = 'Images/'
    modified_images_folder = 'ModifiedImages/'
    augmented_images_folder = 'AugmentedImages/'
    modified_augmented_images_folder = 'ModifiedAugmentedImages/'

    prepare_data(images_folder, modified_images_folder, augmented_images_folder)

    batch_size = 32
    img_height = 180
    img_width = 180
    seed = 123
    validation_split = 0.2

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        modified_images_folder,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        modified_images_folder,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])

    model.compile(optimizer='SGD',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('Executing model...')
    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    print('Model done!')

    results = model.evaluate(train_ds, batch_size=128)

    model.save('LeukemiaModel')

    print('Predicting images...')
    num_img, num_score = 0, 0
    for index, filename in enumerate(os.listdir(modified_augmented_images_folder)):
        img = os.path.join(modified_augmented_images_folder, filename)
        img = kerasImage.load_img(img, target_size=(img_width, img_height))
        img = kerasImage.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        image = Image(filename, img, filename[-5])

        predictions = model.predict(image.img)
        score = tf.nn.softmax(predictions[0])

        if (class_names[np.argmax(score)] == "Leukemia" and image.type == '1') or (
                class_names[np.argmax(score)] == "Normal" and image.type == '0'):
            num_score += 1

        num_img += 1

    os.system('cls' if os.name == 'nt' else 'clear')
    print('\n' * 100)
    print("Test loss: {:.2f}%; Test accuracy: {:.2f}%.".format(results[0] * 100, results[1] * 100))
    print('This model had {:.2f}% accuracy on augmented images.'.format(
        0 if num_score == 0 else num_score * 100 / num_img))