import os
import numpy as np
import cv2

def load_dataset(path):

    images = []
    labels = []

    for filename in os.listdir(path):
        if filename.startswith("cat") and filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(1)

        else:
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(0)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
