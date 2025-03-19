# Exemplar-Based Image Inpainting

An implementation of exemplar-based image inpainting for region filling and object removal in images. This tool allows users to interactively select regions of an image to be removed and automatically fills them with visually plausible content derived from the surrounding areas.

## Overview

This project implements the exemplar-based image inpainting algorithm that uses a patch-based approach to fill in missing or damaged regions in images. The implementation follows the methodology described in the paper "Region Filling and Object Removal by Exemplar-Based Image Inpainting" by Criminisi et al.

The algorithm intelligently propagates texture and structure from the surrounding image into the target region by:
1. Prioritizing patches along the boundary of the missing region
2. Finding the best matching source patches in the image
3. Copying and blending these patches into the target region

## Features

- Interactive GUI for selecting regions to be inpainted
- Priority-based filling mechanism that preserves both structure and texture
- Lab color space comparison for better patch matching
- Confidence and data term calculation for determining fill order
- Visual feedback during the inpainting process

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/exemplar-inpainting.git
cd exemplar-inpainting
pip install -r requirements.txt
```

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-image
- scipy
- OpenCV (cv2)
- PIL/Pillow
- tkinter

## Usage

Run the main script to start the application:

```bash
python __main__.py
```

### Steps to use the application:

1. Click on "open image" to select an image file
2. In the "Mask Builder" window, use the mouse to draw a contour around the area you want to remove
3. Press any key to proceed after finishing the contour
4. The program will display the mask and begin the inpainting process
5. The result will be saved as "output_[original_filename].jpg" in the current directory

## Algorithm Details

The inpainting process follows these steps:

1. **Initialization**: Calculate the confidence values for all pixels (1 for known region, 0 for target region)
2. **Front Detection**: Find the boundary (front) of the target region
3. **Priority Calculation**: For each front pixel, calculate:
   - Confidence term (C): Measure of reliable information around the pixel
   - Data term (D): Strength of isophotes (lines of equal intensity) hitting the front
   - Priority (P): P = C × D
4. **Patch Selection**: Select the patch with the highest priority
5. **Source Patch Search**: Find the best matching patch from the source region
6. **Update**: Copy data from source patch to target patch and update confidence values
7. **Repeat**: Continue until all target pixels are filled

## Implementation Notes

- The implementation uses a patch-based approach with configurable patch size
- Lab color space is used for better perceptual matching between patches
- Euclidean distance is used as a tie-breaker for equally good patch matches
- The algorithm automatically adjusts patch size near image boundaries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation is based on the paper "Region Filling and Object Removal by Exemplar-Based Image Inpainting" by A. Criminisi, P. Perez, and K. Toyama.
- Thanks to the scikit-image and OpenCV communities for providing excellent image processing tools.
