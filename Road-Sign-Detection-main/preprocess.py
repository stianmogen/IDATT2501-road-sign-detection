import os.path

import cv2
import pandas
import pandas as pd
from PIL import Image

input_dir = "dataset/"
preprocessed_dir = "preprocessed/"
train_path = os.path.join(input_dir, "train.csv")
test_path = os.path.join(input_dir, "test.csv")

train_csv = pd.read_csv(train_path)
test_csv = pd.read_csv(test_path)

dim = 112

# Function to read images and crop, resize and applies CLAHE
def preprocess(dataframe):
    img_name = os.path.join(input_dir, dataframe[7])
    image = cv2.imread(img_name)

    # Getting bounding box of image
    x1, y1, x2, y2 = dataframe[2], dataframe[3], dataframe[4], dataframe[5]
    image = image[y1:y2, x1:x2]

    # Resize image to selected size
    image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_LANCZOS4)
    # image = cv2.GaussianBlur(image, (3, 3), 0)

    # Constrast limited adaptive histogram equalization (CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb), dataframe[6]


train_data = pandas.DataFrame(list(map(preprocess, train_csv.values)), columns=["Features", "Labels"])
train_data = train_data[train_data.Labels < 9]

# Split training data into training-validation
train = train_data.sample(frac=0.8, random_state=42)
val = train_data.drop(train.index)

test = pandas.DataFrame(list(map(preprocess, test_csv.values)), columns=["Features", "Labels"])
test = test[test.Labels < 9]  # Only selecting speed limit (class 0 to 8)

if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Writing resized/preprocessed images to file
train.to_pickle(os.path.join(preprocessed_dir, "train.p"))
train.to_pickle(os.path.join(preprocessed_dir, "val.p"))
test.to_pickle(os.path.join(preprocessed_dir, "test.p"))
