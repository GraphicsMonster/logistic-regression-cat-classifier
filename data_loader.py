import numpy as np
import cv2
import os

def load_dataset(path, num_images):
    images = []
    labels = []
    count = 0

    for filename in os.listdir(path):
        
        if count > num_images:
            break

        elif filename.startswith("cat"):

            image = cv2.imread(os.path.join(path, filename))

            if image is not None:

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                images.append(image)
                labels.append(1)
                count += 1
            
            else:
                print("Error: Failed to load File: " + filename)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Ok this works