import numpy as np
from scipy.fftpack import dct, idct

# Function to divide an image into blocks of given size
def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size
    blocks = []

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            y_start = i * block_size
            x_start = j * block_size
            block = image[y_start : y_start + block_size, x_start : x_start + block_size]
            blocks.append(block)

    return blocks

# Function to merge blocks back into a single image
def merge_blocks(dct_blocks, image_shape, block_size):
    num_blocks_height = image_shape[0] // block_size
    num_blocks_width = image_shape[1] // block_size

    merged_dct_blocks = np.zeros((num_blocks_height * block_size, num_blocks_width * block_size), dtype=np.complex128)

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            y_start = i * block_size
            x_start = j * block_size
            merged_dct_blocks[y_start : y_start + block_size, x_start : x_start + block_size] = dct_blocks[i * num_blocks_width + j]

    return merged_dct_blocks.real

# Function to compute the DCT of a block
def dct_block(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

# Function to compute the inverse DCT of a block
def idct_block(dct_block):
    return idct(idct(dct_block.T, norm="ortho").T, norm="ortho")

# Function to compute the DFT of an image using blocks
def my_dft(image, block_size):
    rows, cols = image.shape
    dft_image = np.zeros((rows, cols), dtype=np.complex128)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i : i + block_size, j : j + block_size]
            dft_image[i : i + block_size, j : j + block_size] = compute_dft(block)

    return dft_image

# Function to compute the DFT of a block
def compute_dft(block):
    rows, cols = block.shape
    dft_block = np.zeros((rows, cols), dtype=np.complex128)

    for m in range(rows):
        for n in range(cols):
            sum_val = 0
            for x in range(rows):
                for y in range(cols):
                    sum_val += block[x, y] * np.exp(-2j * np.pi * ((m * x / rows) + (n * y / cols)))
            dft_block[m, n] = sum_val

    return dft_block

# Function to compute the inverse DFT of an image using blocks
def my_idft(dft_image, block_size):
    rows, cols = dft_image.shape
    image = np.zeros((rows, cols), dtype=np.float64)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            dft_block = dft_image[i : i + block_size, j : j + block_size]
            image[i : i + block_size, j : j + block_size] = compute_idft(dft_block)

    return image

# Function to compute the inverse DFT of a block
def compute_idft(dft_block):
    rows, cols = dft_block.shape
    block = np.zeros((rows, cols), dtype=np.float64)

    for x in range(rows):
        for y in range(cols):
            sum_val = 0
            for m in range(rows):
                for n in range(cols):
                    sum_val += dft_block[m, n] * np.exp(2j * np.pi * ((m * x / rows) + (n * y / cols)))
            block[x, y] = sum_val.real / (rows * cols)

    return block

# Function to zero out the last rows and columns of DFT blocks
def zero_last_rows_and_columns(dft_blocks, num_rows_to_zero=4, num_cols_to_zero=4):
    zeroed_dft_blocks = []
    for block in dft_blocks:
        BLOCK_SIZE = block.shape[0]
        zeroed_block = block.copy()
        zeroed_block[BLOCK_SIZE - num_rows_to_zero :, :] = 0
        zeroed_block[:, BLOCK_SIZE - num_cols_to_zero :] = 0
        zeroed_dft_blocks.append(zeroed_block)
    return zeroed_dft_blocks
