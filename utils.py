import numpy as np
from scipy.fftpack import dct, idct


def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size
    blocks = []

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            y_start = i * block_size
            x_start = j * block_size
            block = image[
                y_start : y_start + block_size, x_start : x_start + block_size
            ]
            blocks.append(block)

    return blocks


def merge_blocks(dct_blocks, image_shape, block_size):
    num_blocks_height = image_shape[0] // block_size
    num_blocks_width = image_shape[1] // block_size

    merged_dct_blocks = np.zeros(
        (num_blocks_height * block_size, num_blocks_width * block_size),
        dtype=np.complex128,
    )

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            y_start = i * block_size
            x_start = j * block_size
            merged_dct_blocks[
                y_start : y_start + block_size, x_start : x_start + block_size
            ] = dct_blocks[i * num_blocks_width + j]

    return merged_dct_blocks.real


def dct_block(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct_block(dct_block):
    return idct(idct(dct_block.T, norm="ortho").T, norm="ortho")


def my_dft(image, block_size):
    rows, cols = image.shape
    dft_image = np.zeros((rows, cols), dtype=np.complex128)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i : i + block_size, j : j + block_size]
            dft_image[i : i + block_size, j : j + block_size] = compute_dft(block)

    return dft_image


def compute_dft(block):
    rows, cols = block.shape
    dft_block = np.zeros((rows, cols), dtype=np.complex128)

    for m in range(rows):
        for n in range(cols):
            sum_val = 0
            for x in range(rows):
                for y in range(cols):
                    sum_val += block[x, y] * np.exp(
                        -2j * np.pi * ((m * x / rows) + (n * y / cols))
                    )
            dft_block[m, n] = sum_val

    return dft_block


def my_idft(dft_image, block_size):
    rows, cols = dft_image.shape
    image = np.zeros((rows, cols), dtype=np.float64)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            dft_block = dft_image[i : i + block_size, j : j + block_size]
            image[i : i + block_size, j : j + block_size] = compute_idft(dft_block)

    return image


def compute_idft(dft_block):
    rows, cols = dft_block.shape
    block = np.zeros((rows, cols), dtype=np.float64)

    for x in range(rows):
        for y in range(cols):
            sum_val = 0
            for m in range(rows):
                for n in range(cols):
                    sum_val += dft_block[m, n] * np.exp(
                        2j * np.pi * ((m * x / rows) + (n * y / cols))
                    )
            block[x, y] = sum_val.real / (rows * cols)

    return block


def zero_last_rows_and_columns(dft_blocks, num_rows_to_zero=4, num_cols_to_zero=4):
    zeroed_dft_blocks = []
    for block in dft_blocks:
        BLOCK_SIZE = block.shape[0]
        zeroed_block = block.copy()
        zeroed_block[BLOCK_SIZE - num_rows_to_zero :, :] = 0
        zeroed_block[:, BLOCK_SIZE - num_cols_to_zero :] = 0
        zeroed_dft_blocks.append(zeroed_block)
    return zeroed_dft_blocks


def my_dct2_block(image, block_size):
    rows, cols = image.shape
    dct_matrix = np.zeros((rows, cols), dtype=np.float64)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i + block_size, j:j + block_size]
            dct_block = compute_dct(block, block_size)
            dct_matrix[i:i + block_size, j:j + block_size] = dct_block

    return dct_matrix


def compute_dct(block, block_size):
    rows, cols = block.shape
    dct_block = np.zeros((rows, cols), dtype=np.float64)

    for m in range(rows):
        for n in range(cols):
            sum_val = 0
            for x in range(rows):
                for y in range(cols):
                    sum_val += block[x, y] * \
                               np.cos(np.pi * (2 * x + 1) * m / (2 * rows)) * \
                               np.cos(np.pi * (2 * y + 1) * n / (2 * cols))
            alpha_m = 1 / np.sqrt(rows) if m == 0 else np.sqrt(2) / np.sqrt(rows)
            alpha_n = 1 / np.sqrt(cols) if n == 0 else np.sqrt(2) / np.sqrt(cols)
            dct_block[m, n] = alpha_m * alpha_n * sum_val

    return dct_block


def my_idct2_block(dct_image, block_size):
    rows, cols = dct_image.shape
    inv_matrix = np.zeros((rows, cols), dtype=np.float64)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            dct_block = dct_image[i:i + block_size, j:j + block_size]
            inv_block = compute_idct(dct_block, block_size)
            inv_matrix[i:i + block_size, j:j + block_size] = inv_block

    return inv_matrix


def compute_idct(dct_block, block_size):
    rows, cols = dct_block.shape
    inv_block = np.zeros((rows, cols), dtype=np.float64)

    for x in range(rows):
        for y in range(cols):
            sum_val = 0
            for m in range(rows):
                for n in range(cols):
                    alpha_m = 1 / np.sqrt(rows) if m == 0 else np.sqrt(2) / np.sqrt(rows)
                    alpha_n = 1 / np.sqrt(cols) if n == 0 else np.sqrt(2) / np.sqrt(cols)
                    sum_val += alpha_m * alpha_n * dct_block[m, n] * \
                               np.cos(np.pi * (2 * x + 1) * m / (2 * rows)) * \
                               np.cos(np.pi * (2 * y + 1) * n / (2 * cols))
            inv_block[x, y] = sum_val

    return inv_block
