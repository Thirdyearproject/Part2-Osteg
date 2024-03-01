import cv2
import numpy as np

def divide_image(image, block_size, overlap):
    height, width, _ = image.shape

    step = block_size - overlap

    # Calculate the number of blocks
    num_blocks_x = (width - block_size) // step + 1
    num_blocks_y = (height - block_size) // step + 1

    blocks = []
    for y in range(0, height - block_size + 1, step):
        for x in range(0, width - block_size + 1, step):
            block = image[y:y + block_size, x:x + block_size]
            blocks.append(block)

    return blocks

def calculate_standard_deviation(blocks):
    std_deviations = []
    for block in blocks:
        # Calculate standard deviation for each channel
        std_dev = np.std(block, axis=(0, 1))
        std_deviations.append(std_dev)

    return std_deviations

image = cv2.imread('image.jpg')

# Parameters
block_size = 32  # Size of each block
overlap = 8      # Overlapping pixels between blocks

# Divide image into blocks
blocks = divide_image(image, block_size, overlap)

# Calculate standard deviation for each block
std_deviations = calculate_standard_deviation(blocks)

# Print standard deviations for each block
for i, std_dev in enumerate(std_deviations):
    print(f"Block {i + 1}: Standard Deviation - R: {std_dev[0]}, G: {std_dev[1]}, B: {std_dev[2]}")
