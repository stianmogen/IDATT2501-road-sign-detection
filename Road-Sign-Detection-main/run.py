import cv2
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

# Load speed limit model
model = torch.jit.load("models/model9classes.pt")

# Model image size
dim = 112


def preprocessing(img):
    # Resizing image
    img = cv2.resize(img, (dim, dim), cv2.INTER_LANCZOS4)
    # img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # img = cv2.GaussianBlur(img, (3, 3), 0)

    # Applying CLAHE to all channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Converting image to range 0-1 instead of 0-255
    img = img / 255.

    return img


data = pd.read_csv("signnames.csv")

# Defining width and height to match model dimensions
(width, height) = (dim, dim)

haar_file = "haarcascade.xml"

# Creating cascade classifier
stop_data = cv2.CascadeClassifier(haar_file)
# webcam = cv2.VideoCapture("LINK_TO_VIDEO") # Use this if you want a particular video instead of webcam

webcam = cv2.VideoCapture(0) # If this is selected using default camera on device

softmax = nn.Softmax(dim=1)


while True:
    (_, im) = webcam.read()

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Finding interesting objects
    found = stop_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, maxSize=(200, 200))

    # Loops through all found objects
    for (x, y, w, h) in found:
        # Cropping image by bounding box
        sign = rgb[y:y + h, x:x + w]
        plt.imshow(sign)

        # Preprocessing image
        sign = torch.tensor([preprocessing(sign)])

        # Reshaping dimensions for pytorch model (batchsize, channels, dim1, dim2)
        sign = sign.permute(0, 3, 1, 2).float()
        sign = sign.contiguous()

        # Try to recognize the face
        prediction = model(sign)

        print(prediction)
        item = prediction.argmax(1).cpu().item()
        certainty = softmax(prediction).max()

        # Display green rectangle
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # If the model predicts more than 95% certainty in prediction display name of prediction and metadata
        if certainty > .95:
            cv2.putText(im, '%s' %
                        (data["SignName"][item]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(im, f"{certainty*100}%", (x - 10, y - 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:
        break
