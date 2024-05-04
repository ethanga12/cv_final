import os
import shutil
from random import shuffle

# Define paths
base_dir = '../Desktop/gtzan/1.0.0/images_original'
train_dir = '../Desktop/gtzan/1.0.0/train'
test_dir = '../Desktop/gtzan/1.0.0/test'

# Create training and testing directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Loop through each category in the base directory
for category in os.listdir(base_dir):
    category_dir = os.path.join(base_dir, category)
    
    # Check if it's a directory
    if os.path.isdir(category_dir):
        # Create corresponding directories in train and test
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        if not os.path.exists(train_category_dir):
            os.makedirs(train_category_dir)
        if not os.path.exists(test_category_dir):
            os.makedirs(test_category_dir)
        
        # Get all images and shuffle them
        images = [img for img in os.listdir(category_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        shuffle(images)
        
        # Split images according to the 90-10 ratio
        split_point = int(0.9 * len(images))
        train_images = images[:split_point]
        test_images = images[split_point:]
        
        # Move images to respective directories
        for image in train_images:
            shutil.move(os.path.join(category_dir, image), os.path.join(train_category_dir, image))
        for image in test_images:
            shutil.move(os.path.join(category_dir, image), os.path.join(test_category_dir, image))

print("Data split into training and testing sets successfully.")
