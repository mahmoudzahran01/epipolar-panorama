"""
Image stitching algorithm implementation.
This module includes functions for stitching multiple images together to create panoramic images.
"""

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def est_homography(src, dest):
    """
    Estimate homography matrix from source to destination points
    
    Args:
        src: Source points, shape (N, 2)
        dest: Destination points, shape (N, 2)
        
    Returns:
        Homography matrix H, shape (3, 3)
    """
    N = src.shape[0]
    if N != dest.shape[0]:
        raise ValueError("src and dest should have the same dimension")
    
    src_h = np.hstack((src, np.ones((N, 1))))
    A = np.array([np.block([[src_h[n], np.zeros(3), -dest[n, 0] * src_h[n]],
                            [np.zeros(3), src_h[n], -dest[n, 1] * src_h[n]]])
                  for n in range(N)]).reshape(2 * N, 9)
    
    [_, _, V] = np.linalg.svd(A)
    return V.T[:, 8].reshape(3, 3)

def apply_homography(H, src):
    """
    Apply homography transformation to source points
    
    Args:
        H: Homography matrix, shape (3, 3)
        src: Source points, shape (N, 2)
        
    Returns:
        Transformed points
    """
    src_h = np.hstack((src, np.ones((src.shape[0], 1))))
    dest = src_h @ H.T
    return (dest / dest[:,[2]])[:,0:2]

def getInliers(src, dest, H, threshold):
    """
    Find inliers based on homography transformation
    
    Args:
        src: Source points
        dest: Destination points
        H: Homography matrix
        threshold: Distance threshold for inliers
        
    Returns:
        Indices of inliers
    """
    estimated_dest = apply_homography(H, src)
    tot_dist = np.linalg.norm(dest - estimated_dest, axis=1)
    return np.where(tot_dist < threshold)[0]

def find_homography_ransac(src, dest, threshold=0.5, max_iterations=1000):
    """
    Run RANSAC to estimate homography matrix
    
    Args:
        src, dest: Coordinates
        threshold: Threshold for RANSAC
        max_iterations: Number of iterations for RANSAC
        
    Returns:
        Estimated homography H
    """
    RANSAC_POINT_COUNT = 4
    N = src.shape[0]
    idx_list = list(range(N))
    max_inliers = 0
    best_H = None
    best_inlier_indices = None
    
    for _ in range(max_iterations):
        # 1. Randomly select 4 points
        random_indices = random.sample(idx_list, RANSAC_POINT_COUNT)
        src_subset = src[random_indices, :]
        dest_subset = dest[random_indices, :]
        
        # 2. Compute homography
        H = est_homography(src_subset, dest_subset)

        # 3. Find inliers
        inlier_indices = getInliers(src, dest, H, threshold)
        inlier_count = len(inlier_indices)

        # 4. Update best homography if we found more inliers
        if inlier_count > max_inliers:
            best_H = H
            max_inliers = inlier_count
            best_inlier_indices = inlier_indices
    
    # Recompute homography using all inliers for better accuracy
    if best_inlier_indices is not None and len(best_inlier_indices) >= 4:
        best_H = est_homography(src[best_inlier_indices], dest[best_inlier_indices])
    
    return best_H

def inverse_warp_images(img1, img2, H):
    """
    Warp img2 to align with img1 using homography H
    
    Args:
        img1: First image
        img2: Second image to be warped
        H: Homography matrix
        
    Returns:
        Stitched image
    """
    H_inv = np.linalg.inv(H)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of both images
    im1_corners = np.array([[[0, 0]], [[0, h1]], [[w1, 0]], [[w1, h1]]])
    im2_corners = np.array([[[0, 0]], [[0, h2]], [[w2, 0]], [[w2, h2]]])
    im2_corners_transformed = cv2.perspectiveTransform(im2_corners.astype(np.float32), H_inv)

    # Find min and max coordinates to determine output image size
    stitched_corners = np.vstack((im1_corners, im2_corners_transformed))
    top_left_corner = stitched_corners.min(axis=0).flatten()
    bottom_right_corner = stitched_corners.max(axis=0).flatten()
    xmin, ymin = np.int32(top_left_corner)
    xmax, ymax = np.int32(bottom_right_corner)

    # Create transformation matrix that includes offset
    adjust_coords_mat = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    transform_m = adjust_coords_mat.dot(H_inv)
    new_im_dims = (xmax - xmin, ymax - ymin)

    # Warp img2 into place
    stitched_img = cv2.warpPerspective(img2, transform_m, new_im_dims)
    
    # Place img1 in the correct position
    stitched_img[-ymin:h1-ymin, -xmin:w1-xmin] = img1

    # Crop unnecessary black area
    stitched_img = stitched_img[-ymin:, -xmin:, :]
    
    return stitched_img

def crop_dark(panorama):
    """
    Crop dark areas from panorama
    
    Args:
        panorama: Stitched image
        
    Returns:
        Cropped image
    """
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_area_contour)
    panorama = panorama[y:y+h, x:x+w]

    return panorama

def draw_matches(img1, kp1, img2, kp2, matches):
    """
    Draw feature matches between two images
    
    Args:
        img1, img2: Images
        kp1, kp2: Keypoints for each image
        matches: List of DMatch objects
        
    Returns:
        Image with matches drawn
    """
    keypt1 = [cv2.KeyPoint(coord[1], coord[0], 40) for coord in kp1.tolist()]
    keypt2 = [cv2.KeyPoint(coord[1], coord[0], 40) for coord in kp2.tolist()]
    matches_cv = [cv2.DMatch(pair[0], pair[1], 0) for pair in matches.tolist()]
    return cv2.drawMatches(img1, keypt1, img2, keypt2, matches_cv, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def stitch_images(img1, img2):
    """
    Stitch two images together
    
    Args:
        img1, img2: Images to stitch
        
    Returns:
        Stitched image
    """
    # 1. Detect keypoints
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    ps1 = cv2.KeyPoint_convert(kp1)
    ps2 = cv2.KeyPoint_convert(kp2)

    # 2. Match keypoints
    LOWE_RATIO_THRESH = 0.8
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Filter matches using Lowe's ratio test
    final_matches = []
    for nn_1, nn_2 in matches:
        if nn_1.distance < LOWE_RATIO_THRESH * nn_2.distance:
            final_matches.append(nn_1)

    correspondences = np.array(list(map(lambda m: np.hstack((ps1[m.queryIdx], ps2[m.trainIdx])), final_matches)))

    # 3. Estimate homography with matched keypoints (using RANSAC)
    H = find_homography_ransac(correspondences[:, :2], correspondences[:, 2:])
    H = H / H[2,2]
    
    # 4. Combine images
    stitched_img = inverse_warp_images(img1, img2, H)
    stitched_img = crop_dark(stitched_img)

    return stitched_img

def create_panorama(images):
    """
    Create a panorama from a list of images
    
    Args:
        images: List of images to stitch together
        
    Returns:
        Panoramic image
    """
    if len(images) < 2:
        return images[0] if images else None
        
    # Start with the first image
    panorama = images[0]
    
    # Stitch each subsequent image
    for i in range(1, len(images)):
        panorama = stitch_images(panorama, images[i])
        
    return panorama