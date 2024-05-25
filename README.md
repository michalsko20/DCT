Here's the content for the `README.txt`. You can copy and paste it into your README file:


# Image Compression and Transformation using DCT and DFT

This project contains three main Python scripts for image compression and transformation using Discrete Cosine Transform (DCT). The scripts demonstrate the process of dividing an image into blocks, applying transformations, compressing the transformed data, and reconstructing the image. Additionally, the scripts evaluate the performance and errors associated with the compression methods.

## Files

1. **compression.py**
2. **utils.py**
3. **DCT_alg.py**

## 1. `compression.py`

This script performs compression on an image using DCT with different techniques and visualizes the results.

### Main Steps:
1. **Load and Preprocess Image**:
    - The script loads the image `lena.jpg` and resizes it to 256x256 pixels.
    
2. **Compute DCT for Entire Image**:
    - The DCT of the image is computed and saved.
    
3. **Compression Techniques**:
    - Three compression techniques are applied: 
        1. Thresholding
        2. Block-wise Thresholding
        3. Log Thresholding
        
4. **Reconstruction and Error Calculation**:
    - The image is reconstructed using IDCT and the error is calculated.
    - The number of non-zero DCT coefficients before and after compression is counted and plotted.
    
5. **Plot Results**:
    - The original image, DCT image, reconstructed images, and error images are displayed.
    - The size of the images before and after compression is plotted.

## 2. `utils.py`

This script contains utility functions for DCT, IDCT, DFT, and block manipulation.

### Functions:
- **divide_into_blocks**: Divides an image into non-overlapping blocks.
- **merge_blocks**: Merges blocks back into the image.
- **dct_block**: Computes the DCT for a given block.
- **idct_block**: Computes the IDCT for a given DCT block.
- **zero_last_rows_and_columns**: Zeroes out the last rows and columns of DFT blocks.

## 3. `DCT_alg.py`

This script processes an image using DCT and IDCT with different block sizes and analyzes the time taken and errors.

### Main Steps:
1. **Load and Preprocess Image**:
    - The script loads the image `lena.jpg` and resizes it to 256x256 pixels.
    
2. **Process Image with Different Block Sizes**:
    - The image is divided into blocks of sizes `[4, 8, 16, 32, 64]`.
    - DCT and IDCT are performed on each block size.
    - The DCT and IDCT times are measured and plotted.
    
3. **Plot Results**:
    - The original image, DCT image, and IDCT image are displayed for each block size.
    - The times for DCT and IDCT are plotted against block sizes.
    - Errors are calculated and plotted as Mean Absolute Error (MAE) and Mean Squared Error (MSE).

## Results

- The results for DCT and IDCT analysis are saved in the `DCT_identyfikacja` directory.
- The results for image compression using DCT are saved in the `lena_DCT` directory.

## Dependencies

- OpenCV
- NumPy
- Matplotlib
- SciPy

Install dependencies using:
```bash
pip install opencv-python-headless numpy matplotlib scipy
```

## License

This project is licensed under the MIT License.
```
