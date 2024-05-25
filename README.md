# Image Compression and Transformation using DCT and DFT

This project contains three main Python scripts for image compression and transformation using Discrete Cosine Transform (DCT) and Discrete Fourier Transform (DFT). The scripts demonstrate the process of dividing an image into blocks, applying transformations, compressing the transformed data, and reconstructing the image. Additionally, the scripts evaluate the performance and errors associated with the compression methods.

## Files

1. **lena.py**
2. **utils.py**
3. **DCT_identyfikacja.py**

## 1. `compression.py`

This script demonstrates the process of compressing and reconstructing an image using DCT with different compression methods. It performs the following steps:

- Reads and resizes an image (`lena.jpg`).
- Divides the image into blocks and applies DCT to each block.
- Compresses the DCT coefficients using three different methods:
  - **Compression 1**: Threshold-based compression.
  - **Compression 2**: Block-wise threshold-based compression.
  - **Compression 3**: Logarithmic threshold-based compression.
- Reconstructs the image from the compressed DCT coefficients.
- Calculates and plots the errors between the original and reconstructed images.
- Saves the results and plots to the `lena_DCT` directory.

## 2. `utils.py`

This script contains utility functions used in `lena.py` and `DCT_identyfikacja.py` for dividing images into blocks, merging blocks, and performing DCT and DFT transformations. The main functions are:

- `divide_into_blocks(image, block_size)`: Divides the image into non-overlapping blocks of specified size.
- `merge_blocks(dct_blocks, image_shape, block_size)`: Merges the blocks back into a single image.
- `dct_block(block)`: Computes the DCT of a block.
- `idct_block(dct_block)`: Computes the inverse DCT of a block.
- `my_dft(image, block_size)`: Computes the DFT of an image using blocks.
- `compute_dft(block)`: Computes the DFT of a block.
- `my_idft(dft_image, block_size)`: Computes the inverse DFT of an image using blocks.
- `compute_idft(dft_block)`: Computes the inverse DFT of a block.
- `zero_last_rows_and_columns(dft_blocks, num_rows_to_zero, num_cols_to_zero)`: Zeros out the last specified rows and columns of DFT blocks.

## 3. `DCT_alg.py`

This script demonstrates the process of compressing and reconstructing an image using DFT. It performs the following steps:

- Reads and resizes an image (`lena.jpg`).
- Divides the image into blocks and applies DFT to each block.
- Zeros out the last few rows and columns of the DFT coefficients to simulate compression.
- Reconstructs the image from the modified DFT coefficients.
- Calculates and plots the errors between the original and reconstructed images.
- Measures and plots the time taken to perform DCT and IDCT with different block sizes (8, 16, 32).
- Saves the results and plots to the `DCT_identyfikacja` directory.
