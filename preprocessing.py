from tensorflow import image as tf_image
import numpy as np
from data_loader import load_dataset

def augment_image(image):
    # This function takes an image and returns a new, augmented image
    image = tf_image.random_flip_left_right(image)
    image = tf_image.random_brightness(image, max_delta=0.3)
    image = tf_image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf_image.random_hue(image, max_delta=0.2)
    image = tf_image.random_saturation(image, lower=0.2, upper=1.8)
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
    if len(images.shape) == 3:
        for image in images:
            normalized_image = image / 255.0
            normalized_images.append(normalized_image)
    elif len(images.shape) == 4:
        for image in images:
            normalized_image = image / 255.0
            normalized_images.append(normalized_image)
    else: 
        print("Error: Image shape not supported")

    normalized_images = np.array(normalized_images)
    return normalized_images

