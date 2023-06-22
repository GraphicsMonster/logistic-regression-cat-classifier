import numpy as np
import cv2
import os

def load_dataset(path, num_images=30):
    images = []
    labels = []
    count = 0

    for filename in os.listdir(path):
        if count >= num_images:
            break

        if filename.startswith("cat"):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(1)
                count += 1
            else:
                print('Failed to load image:', img_path)
        else:
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(0)
                count += 1
            else:
                print('Failed to load image:', img_path)

    return np.array(images), np.array(labels)