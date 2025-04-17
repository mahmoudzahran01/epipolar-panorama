"""
Epipolar geometry implementation for computer vision.
This module includes functions for computing the fundamental matrix
using the normalized 8-point algorithm with RANSAC.
"""

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize(x, y):
    """
    Find the transformation T to make coordinates zero mean and the variance as sqrt(2)
    
    Args:
        x, y: Coordinates as arrays
        
    Returns:
        normalized coordinates, transformation T
    """
    points = np.hstack((x, y, np.ones(x.shape)))
    centroid = points.mean(axis=0)
    new_s = np.sqrt(2/np.mean(np.sum((points-centroid)**2, axis=1)))
    T = np.array([[new_s, 0, -new_s*centroid[0]], [0, new_s, -new_s*centroid[1]], [0, 0, 1]])
    res = np.dot(T, points.T).T
    return res, T

def computeF(x1, y1, x2, y2):
    """
    Compute fundamental matrix from corresponding points
    
    Args:
        x1, y1, x2, y2: Coordinates as arrays
        
    Returns:
        fundamental matrix, 3x3
    """
    # Make matrix A using the 8-point algorithm constraint x2^T F x1 = 0
    # where x1 = [x1, y1, 1] and x2 = [x2, y2, 1]
    x1 = x1.flatten()
    y1 = y1.flatten()
    x2 = x2.flatten()
    y2 = y2.flatten()
    
    # Create the constraint matrix
    A = np.zeros((len(x1), 9))
    for i in range(len(x1)):
        A[i] = [x1[i]*x2[i], x1[i]*y2[i], x1[i], 
                y1[i]*x2[i], y1[i]*y2[i], y1[i], 
                x2[i], y2[i], 1]
    
    # Solve for F using SVD
    U, S, V = np.linalg.svd(A)
    
    # Get the solution (last column of V)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint on F
    U_f, S_f, V_f = np.linalg.svd(F)
    S_f[-1] = 0
    F = U_f @ np.diag(S_f) @ V_f
    
    return F

def getInliers(x1, y1, x2, y2, F, thresh):
    """
    Implement the criteria for checking inliers
    
    Args:
        x1, y1, x2, y2: Coordinates as arrays
        F: Estimated fundamental matrix, 3x3
        thresh: Threshold for passing the error
        
    Returns:
        Indices of inliers
    """
    # Construct point arrays
    points1 = np.column_stack((x1, y1, np.ones(x1.shape)))
    points2 = np.column_stack((x2, y2, np.ones(x2.shape)))
    
    # Compute epipolar lines in second image
    # l' = F·x
    lines2 = np.dot(F, points1.T).T
    
    # Compute epipolar lines in first image
    # l = F^T·x'
    lines1 = np.dot(F.T, points2.T).T
    
    # Calculate distances from points to their corresponding epipolar lines
    # Distance from point x to line l: d = |x·l| / sqrt(l[0]^2 + l[1]^2)
    
    # Point-to-line distances in the first image
    dist1 = np.abs(np.sum(points1 * lines1, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    
    # Point-to-line distances in the second image
    dist2 = np.abs(np.sum(points2 * lines2, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    
    # Total distance (Sampson distance)
    total_dist = dist1 + dist2
    
    # Find indices of inliers (points with distance less than threshold)
    inliers = np.where(total_dist < thresh)[0]
    
    return inliers

def ransacF(x1, y1, x2, y2, num_iterations=1000, threshold=0.01):
    """
    Run RANSAC to estimate F
    
    Args:
        x1, y1, x2, y2: Coordinates as arrays
        num_iterations: Number of RANSAC iterations
        threshold: Threshold for inlier check
        
    Returns:
        Best fundamental matrix, corresponding inlier indices
    """
    RANSAC_POINT_COUNT = 8
    num_points = len(x1)
    idx_list = list(range(num_points))
    max_inliers = 0
    best_F = None
    best_inlier_indices = None
    
    for _ in range(num_iterations):
        # 1. Randomly select 8 points
        random_indices = random.sample(idx_list, RANSAC_POINT_COUNT)
        
        # 2. Compute F from those 8 points
        F = computeF(x1[random_indices], y1[random_indices], x2[random_indices], y2[random_indices])
        
        # 3. Find inliers
        inlier_indices = getInliers(x1, y1, x2, y2, F, threshold)
        inlier_count = len(inlier_indices)
        
        # 4. Update best F if we found more inliers
        if inlier_count > max_inliers:
            best_F = F
            max_inliers = inlier_count
            best_inlier_indices = inlier_indices
    
    # Recompute F using all inliers for more accuracy
    if best_inlier_indices is not None and len(best_inlier_indices) >= 8:
        best_F = computeF(x1[best_inlier_indices], y1[best_inlier_indices], 
                          x2[best_inlier_indices], y2[best_inlier_indices])
    
    return best_F, best_inlier_indices

def estimate_fundamental_matrix(x1, y1, x2, y2, num_iterations=1000, threshold=0.01):
    """
    Estimate fundamental matrix from point correspondences using the normalized 8-point algorithm with RANSAC
    
    Args:
        x1, y1: Coordinates in first image
        x2, y2: Coordinates in second image
        num_iterations: Number of RANSAC iterations
        threshold: Threshold for inlier check
        
    Returns:
        Normalized fundamental matrix F
    """
    # 1. Normalize coordinates
    norm_p1, T1 = normalize(x1.reshape(-1, 1), y1.reshape(-1, 1))
    norm_x1, norm_y1 = norm_p1[:, 0], norm_p1[:, 1]
    norm_p2, T2 = normalize(x2.reshape(-1, 1), y2.reshape(-1, 1))
    norm_x2, norm_y2 = norm_p2[:, 0], norm_p2[:, 1]

    # 2. Run 8-point algorithm with RANSAC to estimate F
    F, _ = ransacF(norm_x1, norm_y1, norm_x2, norm_y2, num_iterations=num_iterations, threshold=threshold)

    # 3. De-normalize F
    F = T2.T @ F @ T1

    # 4. Normalize F so that the last entry is 1
    F = F / F[2, 2]
    
    return F

def plot_epipolar_lines(img1, img2, x1, y1, x2, y2, F, num_points=7):
    """
    Plot epipolar lines for point correspondences
    
    Args:
        img1, img2: Images to plot on
        x1, y1, x2, y2: Point coordinates
        F: Fundamental matrix
        num_points: Number of points to sample
    
    Returns:
        Image with epipolar lines
    """
    h1, w1, nc1 = img1.shape
    h2, w2, nc2 = img2.shape

    assert nc1 == nc2

    # Sample random points
    n = len(x1)
    idx_list = list(range(n))
    random_indices = random.sample(idx_list, min(num_points, n))
    
    # Make sure the arrays are properly shaped for stacking
    x1_subset = np.array(x1[random_indices]).reshape(1, -1)
    y1_subset = np.array(y1[random_indices]).reshape(1, -1)
    x2_subset = np.array(x2[random_indices]).reshape(1, -1)
    y2_subset = np.array(y2[random_indices]).reshape(1, -1)
    ones = np.ones((1, len(random_indices)))
    
    # Create homogeneous coordinates
    points1 = np.vstack((x1_subset, y1_subset, ones))
    points2 = np.vstack((x2_subset, y2_subset, ones))

    # Compute epipolar lines
    lines1 = np.dot(F.T, points2)
    lines2 = np.dot(F, points1)

    p1s = points1[:2, :].astype(np.int32)
    p2s = points2[:2, :].astype(np.int32)

    w = img1.shape[1]
    GREEN_COLOR = (0,255,0)
    RED_COLOR = (0,0,255)
    radius = 3
    
    # Create copies of the images to draw on
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    for i in range(len(random_indices)):
        # Draw epipolar line in the first image
        a, b, c = lines1[:,i]
        x1_line, y1_line = 0, int(-c/b) if b != 0 else 0
        x2_line, y2_line = w, int(-a/b * w - c/b) if b != 0 else 0
        cv2.line(img1_copy, (x1_line, y1_line), (x2_line, y2_line), GREEN_COLOR)
        
        # Draw point in the first image
        pt_x, pt_y = p1s[:, i]
        cv2.circle(img1_copy, (pt_x, pt_y), radius, RED_COLOR, 1)

        # Draw epipolar line in the second image
        a, b, c = lines2[:,i]
        x1_line, y1_line = 0, int(-c/b) if b != 0 else 0
        x2_line, y2_line = w, int(-a/b * w - c/b) if b != 0 else 0
        cv2.line(img2_copy, (x1_line, y1_line), (x2_line, y2_line), GREEN_COLOR)
        
        # Draw point in the second image
        pt_x, pt_y = p2s[:, i]
        cv2.circle(img2_copy, (pt_x, pt_y), radius, RED_COLOR, 1)

    # Concatenate images side by side
    concatenated_img = np.zeros((h1, w1+w2, nc1), dtype=np.uint8)
    concatenated_img[0:h1, 0:w1] = img1_copy
    concatenated_img[0:h1, w1:w1+w2] = img2_copy
    
    return concatenated_img