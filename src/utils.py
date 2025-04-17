"""
Utility functions for image processing, visualization, and data handling.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def show_image(img, scale=1.0):
    """
    Display an image using matplotlib
    
    Args:
        img: Image to display
        scale: Scale factor for figure size
    """
    plt.figure(figsize=scale * plt.figaspect(1))
    plt.imshow(img, interpolation='nearest')
    plt.gray()
    plt.axis('off')
    plt.show()
    
def display_image(img):
    """
    A replacement for cv2_imshow that works in any environment
    
    Args:
        img: Image to display
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
    else:
        # For grayscale images
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def load_data_from_mat(file_path):
    """
    Load matched point data from .mat file
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        x1, y1, x2, y2: Arrays of corresponding points
    """
    from scipy.io import loadmat
    
    data = loadmat(file_path)
    r1 = data['r1']
    r2 = data['r2']
    c1 = data['c1']
    c2 = data['c2']
    matches = data['matches']
    
    # Extract matched keypoints
    x1 = c1[matches[:,0]-1]
    y1 = r1[matches[:,0]-1]
    x2 = c2[matches[:,1]-1]
    y2 = r2[matches[:,1]-1]
    
    return x1, y1, x2, y2

def load_images_from_directory(directory):
    """
    Load all images from a directory
    
    Args:
        directory: Directory path
        
    Returns:
        List of loaded images
    """
    import os
    import glob
    
    img_list = sorted(glob.glob(os.path.join(directory, "*.JPG"))) + \
               sorted(glob.glob(os.path.join(directory, "*.jpg"))) + \
               sorted(glob.glob(os.path.join(directory, "*.png")))
    
    images = []
    for img_path in img_list:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load image {img_path}")
    
    return images

def plot_matches(image1, kp1, image2, kp2, matches):
    """
    Draw matches between two images
    
    Args:
        image1, image2: Images
        kp1, kp2: Keypoints
        matches: Matches between keypoints
        
    Returns:
        Image with matches drawn
    """
    # Convert keypoints to OpenCV format
    keypt1 = [cv2.KeyPoint(coord[1], coord[0], 40) for coord in kp1.tolist()]
    keypt2 = [cv2.KeyPoint(coord[1], coord[0], 40) for coord in kp2.tolist()]
    
    # Convert matches to OpenCV format
    cv_matches = [cv2.DMatch(pair[0], pair[1], 0) for pair in matches.tolist()]
    
    # Draw matches
    matched_img = cv2.drawMatches(image1, keypt1, image2, keypt2, cv_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    return matched_img