import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct, idct

# Create the output directory if it doesn't exist
output_dir = "lena_DCT"
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess the image
image = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

# Save the original image
cv2.imwrite(os.path.join(output_dir, "original_image.jpg"), image)


# Function to count non-zero elements in a matrix
def count_nonzero_elements(matrix):
    return np.count_nonzero(matrix)


# Compression functions
def compression(dct_image, compression_ratio):
    flattened = np.abs(dct_image.flatten())
    threshold = np.sort(flattened)[-int(compression_ratio * len(flattened))]
    compressed_dct = dct_image * (np.abs(dct_image) > threshold)
    return compressed_dct


def compression2(dct_image, block_size, compression_ratio):
    h, w = dct_image.shape
    compressed_dct = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_image[i : i + block_size, j : j + block_size]
            compressed_dct[i : i + block_size, j : j + block_size] = compression(
                block, compression_ratio
            )
    return compressed_dct


def compression3(dct_image, threshold):
    compressed_dct = dct_image * (np.log1p(np.abs(dct_image)) >= threshold)
    return compressed_dct


# Function to visualize and save results
def plot_and_save_results(image, dct_image, compressed_dct, title, filename):
    reconstructed_image = np.zeros_like(image)
    h, w = dct_image.shape
    block_size = 8

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = compressed_dct[i : i + block_size, j : j + block_size]
            reconstructed_block = idct(idct(block.T, norm="ortho").T, norm="ortho")
            reconstructed_image[i : i + block_size, j : j + block_size] = (
                reconstructed_block
            )

    error = np.abs(image - reconstructed_image)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(np.log1p(np.abs(dct_image)), cmap="gray")
    plt.title("DCT Coefficients")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error, cmap="gray")
    plt.title("Error")
    plt.axis("off")

    plt.suptitle(title)
    plt.savefig(os.path.join(output_dir, filename), format="jpg")
    plt.close()  # Close the plot after saving

    return reconstructed_image


# Function to display the number of non-zero coefficients
def plot_nonzero_counts(original, compressed1, compressed2, compressed3, filename):
    counts = [original, compressed1, compressed2, compressed3]
    labels = ["Original", "Compression 1", "Compression 2", "Compression 3"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=["blue", "orange", "green", "red"])
    plt.title("Non-zero DCT Coefficients Before and After Compression")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, filename), format="jpg")
    plt.close()  # Close the plot after saving


# Compute the DCT for the entire image
def compute_dct_image(image):
    dct_image = np.zeros_like(image, dtype=np.float32)
    BLOCK_SIZE = 8
    for i in range(0, image.shape[0], BLOCK_SIZE):
        for j in range(0, image.shape[1], BLOCK_SIZE):
            block = image[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE].astype(np.float32)
            dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
            dct_image[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE] = dct_block
    return dct_image


dct_image = compute_dct_image(image)

# Save the DCT image
cv2.imwrite(
    os.path.join(output_dir, "dct_image.jpg"),
    (dct_image / np.max(dct_image) * 255).astype(np.uint8),
)

# Compression 1
compression_ratio = 0.1
compressed_dct1 = compression(dct_image, compression_ratio)
reconstructed_image1 = plot_and_save_results(
    image, dct_image, compressed_dct1, "Compression 1: Thresholding", "compression1.jpg"
)

# Compression 2
block_size = 8
compressed_dct2 = compression2(dct_image, block_size, compression_ratio)
reconstructed_image2 = plot_and_save_results(
    image,
    dct_image,
    compressed_dct2,
    "Compression 2: Block-wise Thresholding",
    "compression2.jpg",
)

# Compression 3
threshold = 4.0
compressed_dct3 = compression3(dct_image, threshold)
reconstructed_image3 = plot_and_save_results(
    image,
    dct_image,
    compressed_dct3,
    "Compression 3: Log Thresholding",
    "compression3.jpg",
)

# Original DCT and IDCT for comparison
idct_image = np.zeros_like(image, dtype=np.float32)
BLOCK_SIZE = 8
for i in range(0, dct_image.shape[0], BLOCK_SIZE):
    for j in range(0, dct_image.shape[1], BLOCK_SIZE):
        block = dct_image[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
        reconstructed_block = idct(idct(block.T, norm="ortho").T, norm="ortho")
        idct_image[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE] = reconstructed_block

# Save the IDCT image
cv2.imwrite(
    os.path.join(output_dir, "idct_image.jpg"),
    (idct_image / np.max(idct_image) * 255).astype(np.uint8),
)

# Display the original image, DCT image, and IDCT image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
p
