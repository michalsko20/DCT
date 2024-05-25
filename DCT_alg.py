import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import os

# Create the output directory if it doesn't exist
output_dir = "DCT_identyfikacja"
os.makedirs(output_dir, exist_ok=True)


# Function to process the image with different block sizes
def process_image_with_different_block_sizes(image, block_sizes, output_dir):
    fig, axs = plt.subplots(len(block_sizes), 3, figsize=(15, 5 * len(block_sizes)))
    dct_times = []
    idct_times = []

    for idx, block_size in enumerate(block_sizes):
        # Compute DCT for the image with the given block size
        start_time = time.time()
        dct_image = utils.my_dct2_block(image, block_size)
        dct_time = time.time() - start_time
        dct_times.append(dct_time)

        # Compute IDCT for the image with the given block size
        start_time = time.time()
        idct_image = utils.my_idct2_block(dct_image, block_size)
        idct_time = time.time() - start_time
        idct_times.append(idct_time)

        # Display results
        axs[idx, 0].imshow(image, cmap="gray")
        axs[idx, 0].set_title(f"Original Image (Block Size: {block_size})")
        axs[idx, 0].axis("off")

        axs[idx, 1].imshow(np.log1p(np.abs(dct_image)), cmap="gray")
        axs[idx, 1].set_title("DCT of Image")
        axs[idx, 1].axis("off")

        axs[idx, 2].imshow(idct_image, cmap="gray")
        axs[idx, 2].set_title("IDCT of Image")
        axs[idx, 2].axis("off")
        print("dzialam")
    plt.suptitle("DCT and IDCT for Different Block Sizes")
    plt.savefig(
        os.path.join(output_dir, "DCT_IDCT_Different_Block_Sizes.jpg"), format="jpg"
    )
    plt.show()

    return dct_times, idct_times


# Load and preprocess the image
BLOCK_SIZE = 8
image = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

# Block sizes to test
block_sizes = [2, 4, 8, 16]

# Process the image for different block sizes
dct_times, idct_times = process_image_with_different_block_sizes(
    image, block_sizes, output_dir
)

# Plot DCT and IDCT times
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(block_sizes, dct_times, marker="o", label="DCT Time")
plt.xlabel("Block Size")
plt.ylabel("Time (seconds)")
plt.title("DCT Time vs Block Size")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(block_sizes, idct_times, marker="o", label="IDCT Time")
plt.xlabel("Block Size")
plt.ylabel("Time (seconds)")
plt.title("IDCT Time vs Block Size")
plt.grid(True)

plt.suptitle("Time Analysis for DCT and IDCT")
plt.savefig(os.path.join(output_dir, "Time_Analysis_DCT_IDCT.jpg"), format="jpg")
plt.show()

# Print DCT and IDCT times
for idx, block_size in enumerate(block_sizes):
    print(f"Block Size: {block_size}")
    print(f"DCT Time: {dct_times[idx]:.4f} seconds")
    print(f"IDCT Time: {idct_times[idx]:.4f} seconds\n")

# Divide the image into blocks
blocks = utils.divide_into_blocks(image, BLOCK_SIZE)

# Merge blocks into one image
merged_blocks = utils.merge_blocks(blocks, image.shape, BLOCK_SIZE)

# Calculate DCT for each block
dct_blocks = [utils.my_dct2_block(block, BLOCK_SIZE) for block in blocks]

# Prepare plots
fig, axs = plt.subplots(3, 8, figsize=(20, 10))

mae_values = []
mse_values = []
percent_mae_values = []
percent_mse_values = []

for i in range(1, 9):
    zeroed_dct_blocks = utils.zero_last_rows_and_columns(dct_blocks, i, i)
    merged_blocks_dct = utils.merge_blocks(zeroed_dct_blocks, image.shape, BLOCK_SIZE)

    axs[0, i - 1].imshow(np.log1p(np.abs(merged_blocks_dct)), cmap="gray")
    axs[0, i - 1].axis("off")
    axs[0, i - 1].set_title(f"Zero last {i} rows/cols")

    idct_blocks = [
        utils.my_idct2_block(dct_block_result, BLOCK_SIZE)
        for dct_block_result in zeroed_dct_blocks
    ]
    merged_idct_blocks = utils.merge_blocks(idct_blocks, image.shape, BLOCK_SIZE)

    axs[1, i - 1].imshow(merged_idct_blocks, cmap="gray")
    axs[1, i - 1].axis("off")
    axs[1, i - 1].set_title(f"Reconstructed {i}")

    error = np.abs(merged_idct_blocks.astype(float) - merged_blocks.astype(float))
    axs[2, i - 1].imshow(error, cmap="gray")
    axs[2, i - 1].axis("off")
    axs[2, i - 1].set_title(f"Error {i}")

    # Calculate error metrics
    mae = np.mean(error)
    mse = np.mean(error**2)
    mae_values.append(mae)
    mse_values.append(mse)

    # Calculate percentage errors
    percent_mae = (mae / np.mean(merged_blocks)) * 100
    percent_mse = (mse / np.mean(merged_blocks**2)) * 100
    percent_mae_values.append(percent_mae)
    percent_mse_values.append(percent_mse)

plt.suptitle("DCT and IDCT with Varying Zeroed Rows and Columns")
plt.savefig(os.path.join(output_dir, "DCT_IDCT_Error_subplot.jpg"), format="jpg")
plt.show()

# Plot percentage MAE and MSE values
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 9), percent_mae_values, marker="o")
plt.title("Mean Absolute Error (MAE) [%]")
plt.xlabel("Number of Zeroed Rows/Cols")
plt.ylabel("MAE [%]")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 9), percent_mse_values, marker="o")
plt.title("Mean Squared Error (MSE) [%]")
plt.xlabel("Number of Zeroed Rows/Cols")
plt.ylabel("MSE [%]")
plt.grid(True)

plt.suptitle("Error Metrics (Percentage)")
plt.savefig(os.path.join(output_dir, "error_metrics_percentage.jpg"), format="jpg")
plt.show()

# Print error values
for i in range(1, 9):
    print(f"Zero last {i} rows/cols:")
    print(f"MAE: {mae_values[i-1]:.2f}, MSE: {mse_values[i-1]:.2f}")
    print(
        f"MAE (%): {percent_mae_values[i-1]:.2f}%, MSE (%): {percent_mse_values[i-1]:.2f}%\n"
    )
