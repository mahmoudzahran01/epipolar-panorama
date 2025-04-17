# Epipolar Panorama

This repository contains implementations of two computer vision algorithms:

1. **Epipolar Geometry**: Implementation of the normalized 8-point algorithm with RANSAC for automatically estimating the fundamental matrix.
2. **Image Stitching**: Algorithm for automatically stitching images together to create panoramic views.

## Example Results

### Epipolar Geometry
![Epipolar Lines](results/epi.jpg)

The green lines show the epipolar lines for the corresponding points (marked in red). Notice how these points follow the epipolar constraint.

### Image Stitching Results

#### Hill Panorama
**Input**: Images in `data/hill/`  
**Output**: [Hill Panorama](results/hill.jpg)

#### TV Panorama
**Input**: Images in `data/tv/`  
**Output**: [TV Panorama](results/tv.jpg)

## Project Structure

```
.
├── data/                 # Data directory
│   ├── Part1_data/       # Data for epipolar geometry
│   │   ├── chapel00.png
│   │   ├── chapel01.png
│   │   └── matches.mat
│   ├── hill/             # Hill panorama input images
│   └── tv/               # TV panorama input images
├── results/              # Results directory
│   ├── epi.jpg           # Epipolar geometry visualization
│   ├── hill.jpg          # Hill panorama result
│   └── tv.jpg            # TV panorama result
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── __init__.py
│   ├── epipolar_geometry.py
│   ├── image_stitching.py
│   └── utils.py
└── requirements.txt      # Python dependencies
```

## Setup and Installation

### Option 1: Using pip

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/epipolar-panorama.git
   cd epipolar-panorama
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Option 2: Using conda

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/epipolar-panorama.git
   cd epipolar-panorama
   ```

2. Create a conda environment:
   ```
   conda create -n epipolar python=3.10
   conda activate epipolar
   ```

3. Install dependencies:
   ```
   conda install -c conda-forge numpy scipy matplotlib opencv
   conda install -c conda-forge jupyter scikit-learn
   ```

4. Register the environment with Jupyter:
   ```
   python -m ipykernel install --user --name=epipolar --display-name="Python (epipolar)"
   ```

## Running the Code

### Using Jupyter Notebook

The easiest way to run the code is through the provided Jupyter notebook:

```bash
jupyter notebook notebooks/main_notebook.ipynb
```

If using conda, make sure to select the "Python (epipolar)" kernel.

### Running From Command Line

To run a quick test of the main functionality:

```python
import os
import cv2
import numpy as np
from src.epipolar_geometry import estimate_fundamental_matrix, plot_epipolar_lines
from src.image_stitching import stitch_images
from src.utils import load_data_from_mat, load_images_from_directory

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Test epipolar geometry
data_path = './data/Part1_data/matches.mat'
x1, y1, x2, y2 = load_data_from_mat(data_path)
F = estimate_fundamental_matrix(x1, y1, x2, y2)

# Load chapel images
img1 = cv2.imread('./data/Part1_data/chapel00.png')
img2 = cv2.imread('./data/Part1_data/chapel01.png')
result_img = plot_epipolar_lines(img1, img2, x1, y1, x2, y2, F, num_points=7)
cv2.imwrite('results/epi.jpg', result_img)

# Test image stitching with hill dataset
hill_images = load_images_from_directory('./data/hill')
stitched_23 = stitch_images(hill_images[1], hill_images[2])
final_hill_panorama = stitch_images(hill_images[0], stitched_23)
cv2.imwrite('results/hill.jpg', final_hill_panorama)

# Test image stitching with tv dataset
tv_images = load_images_from_directory('./data/tv')
stitched_23 = stitch_images(tv_images[1], tv_images[2])
final_tv_panorama = stitch_images(tv_images[0], stitched_23)
cv2.imwrite('results/tv.jpg', final_tv_panorama)

print("Testing complete! Check the results directory for output images.")
```

## Features

### Epipolar Geometry
- Epipolar line visualization
- Normalized 8-point algorithm
- RANSAC for robust estimation

### Image Stitching
- Feature detection and matching
- Homography estimation with RANSAC
- Image warping and blending

## License

MIT License