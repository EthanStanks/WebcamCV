import tensorflow as tf
from keras import layers, models
from keras.utils import load_img, img_to_array
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

EPOCHS = 10
IMG_SIZE = (128, 128)

def process_image(filepath, img_size):
    # loads the image from the file path passed in and then normalizes it
    image = load_img(filepath, target_size=img_size, color_mode='rgb')
    return img_to_array(image) / 255.0

def load_and_preprocess_dataset():
    # Load data from CSV file containing labels and file paths to images
    path = os.path.join("data/","cards.csv")
    df = pd.read_csv(path)

    images = []
    labels = []
    rows = len(df)

    # Loops through each row adding the image and labels to the containers
    print("Loading Images and Labels...")
    for i, row in df.iterrows():
        print(f"Loading {i}/{rows}")
        image_path = os.path.join("data/", row['filepaths'])
        if os.path.exists(image_path):
            images.append(process_image(image_path, IMG_SIZE))
            labels.append(row['labels'])
        else:
            print(f"Image file path doesn't exist for {image_path}")
    print("Completed Loading Images and Labels.")

    images = np.array(images)
    labels = np.array(labels)

    # Data came split up already so index how they had it
    train_idx = df['data set'] == 'train'
    valid_idx = df['data set'] == 'valid'
    test_idx = df['data set'] == 'test'

    train_images, train_labels = images[train_idx], labels[train_idx]
    valid_images, valid_labels = images[valid_idx], labels[valid_idx]
    test_images, test_labels = images[test_idx], labels[test_idx]

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def build_model():
    pass
        
def compile_model(model):
    pass
    
def train_model(model):
    pass

if __name__ == '__main__':
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_and_preprocess_dataset()