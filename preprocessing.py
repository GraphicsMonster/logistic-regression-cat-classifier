import tensorflow as tf
import numpy as np
from data_loader import load_dataset

def augment_image(image):
    # This function takes an image and returns a new, augmented image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.2, upper=1.8)
    return image

def augment_dataset(images, labels):
    # This function takes a dataset and returns a new, augmented dataset
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
       augmented_image = augment_image(image)
       augmented_images.append(augmented_image)
       augmented_labels.append(label)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

def normalize_data(images):
    # This function takes a dataset and returns a new, normalized dataset
    normalized_images = []
    for image in images:
        normalized_image = tf.image.per_image_standardization(image)
        normalized_images.append(normalized_image)
    normalized_images = np.array(normalized_images)
    return normalized_images
