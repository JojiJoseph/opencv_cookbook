import cv2
import numpy as np
import os
from tqdm import tqdm
import random

# Please download Pins Face Recognition from https://www.kaggle.com/hereisburak/pins-face-recognition and copy the folder
# 105_classes_pins_dataset into this folder

# Create recognizer object
recognizer = cv2.face.EigenFaceRecognizer_create()

# Setup dataset
classes = (os.listdir("105_classes_pins_dataset"))
random.shuffle(classes)
labels = []
train_set = []
test_set = []

for i, class_ in enumerate(classes):
    if i == 4:
        break
    images = os.listdir(os.path.join("105_classes_pins_dataset", class_))
    train_set.append(images[:80])
    test_set.append(images[80:100])
    labels.append(class_[5:])

# Training
int_labels = []
images = []
for i, image_paths in tqdm(enumerate(train_set), total=len(train_set)):
    if i == 4:
        break
    for path in image_paths:
        path = os.path.join("105_classes_pins_dataset",
                            "pins_" + labels[i], path)
        img = cv2.imread(path, 0)
        images.append(
            cv2.resize(img, (128, 128))
        )
        int_labels.append(i)

print("Training..")
recognizer.train(images, np.array(int_labels))
print("Training is over!")

# Testing
n_success, n_total = 0, 0
for i, image_paths in tqdm(enumerate(test_set), total=len(test_set)):

    if i == 4:
        break
    for path in image_paths:
        path = os.path.join("105_classes_pins_dataset",
                            "pins_" + labels[i], path)
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (128, 128))

        class_, _ = recognizer.predict(img)

        if class_ == i:
            n_success += 1
        n_total += 1


print("Test accuracy", n_success/n_total * 100)

# Testing visually
for i in range(4):
    path = test_set[i][np.random.choice(list(range(len(test_set[i]))))]
    path = os.path.join("105_classes_pins_dataset", "pins_" + labels[i], path)

    img_org = cv2.imread(path)
    if img_org.shape[1] < 400:
        img_org = cv2.resize(
            img_org, (400, int(400/img_org.shape[1]*img_org.shape[0])))
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (128, 128))
    prediction, _ = recognizer.predict(img_resized)

    cv2.putText(img_org, "Actual: " +
                labels[i], (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
    cv2.putText(img_org, "Predicted: " +
                labels[prediction], (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
    cv2.imshow(labels[i], img_org)
    cv2.waitKey()
    cv2.destroyAllWindows()
