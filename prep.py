import numpy as np
from PIL import Image

def calculate_std_dev(image, block_size):
    """Calculates the standard deviation of each block in an image."""

    image_array = np.array(image, dtype=np.float32)
    height, width = image_array.shape[:2]

    # Calculate step size for overlapping blocks
    step = block_size // 2

    # Initialize empty array to store standard deviation values
    std_dev_image = np.zeros((height, width), dtype=np.float32)

    # Iterate through each block
    for y in range(0, height - block_size + 1, step):
        for x in range(0, width - block_size + 1, step):
            # Extract block
            block = image_array[y:y+block_size, x:x+block_size]

            # Calculate standard deviation (vectorized for efficiency)
            std_dev = np.std(block)

            # Assign standard deviation to corresponding region in std_dev_image
            std_dev_image[y:y+block_size, x:x+block_size] = std_dev
            #print("Standard deviation of block at (", x, ",", y, "):", std_dev)

    return std_dev_image

def otsu_threshold(std_dev_image):
    """Calculates the Otsu threshold for an image."""

    # Flatten the standard deviation image into a 1D array
    std_dev_values = std_dev_image.flatten()

    # Calculate the histogram of standard deviation values
    hist = np.histogram(std_dev_values, bins=256)[0]

    # Normalize the histogram
    hist = hist.astype(float) / std_dev_values.size

    # Initialize variables for Otsu's method
    best_threshold = 0
    max_variance = 0

    for t in range(1, 256):  # Start from 1, not 0 (consistent with binning)
        # Separate pixels into two classes based on the threshold
        w0 = np.sum(hist[:t])
        w1 = 1 - w0

        # Calculate mean of each class (simplified)
        u0 = np.sum(np.arange(t) * hist[:t]) / w0 if w0 > 0 else 0
        u1 = np.sum(np.arange(t, 256) * hist[t:]) / w1 if w1 > 0 else 0

        # Calculate between-class variance
        variance = w0 * w1 * (u0 - u1) ** 2

        # Update best threshold if current variance is higher
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    # The best threshold found by Otsu's method is our ξ
    xi = best_threshold
    print(best_threshold)
    return xi
    

def fake_embedding(image_array, witness_image):
    # Simulate embedding by randomly altering some pixel values
    # based on the witness image.
    stego_array = image_array.copy()
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            if witness_image[y, x] == 255:  # If block is a candidate
                if np.random.rand() < 0.5:  # Randomly modify with 50% chance
                    stego_array[y, x] += 1  # Increase pixel value by 1
    return stego_array

def recursive_otsu_thresholding(image, block_size, epsilon=0.1, max_iterations=10):
    # Initial Otsu threshold and alpha
    xi = otsu_threshold(calculate_std_dev(image, block_size))
    alpha = 1

    for k in range(max_iterations):
        threshold = alpha * xi

        # Create witness images for cover and stego
        witness_cover = np.zeros_like(image, dtype=np.uint8)
        witness_stego = np.zeros_like(image, dtype=np.uint8)

        witness_cover[calculate_std_dev(image, block_size) > threshold] = 255

        # Fake embedding
        stego_image_array = fake_embedding(np.array(image), witness_cover)
        witness_stego[calculate_std_dev(Image.fromarray(stego_image_array), block_size) > threshold] = 255


        # Compare witness images
        if np.array_equal(witness_cover, witness_stego):
            return alpha  # Optimal alpha found

        # Update alpha for next iteration
        alpha += k * epsilon

    return None  # Optimal alpha not found within iterations


# Load the image (ensure grayscale conversion if needed)
image_path = 'image.jpg'
image = Image.open(image_path).convert('L')

# Define block size (consider making it configurable)
block_size = 8

# Calculate standard deviation image
std_dev_image = calculate_std_dev(image, block_size)

# Display the resulting standard deviation image (optional)
std_dev_image_display = Image.fromarray((std_dev_image * 255).astype(np.uint8))
std_dev_image_display.show()

# Apply Otsu's method to get the fixed threshold ξ
xi = otsu_threshold(std_dev_image)

# Set the changeable factor α (optional)
alpha = 1

# Calculate the final threshold
threshold = alpha * xi
print("Threshold:", threshold)

# Form the two classes of blocks
mask = std_dev_image > threshold
class1_blocks = std_dev_image[~mask]
class2_blocks = std_dev_image[mask]
#print(class2_blocks.size) 

witness_image = np.zeros_like(std_dev_image, dtype=np.uint8)
witness_image[std_dev_image > threshold] = 255  # White for candidate blocks

# Display the witness image (optional)
witness_image_display = Image.fromarray(witness_image)
witness_image_display.show()

hist = np.histogram(std_dev_image, bins=256)[0]

# Normalize the histogram to get probabilities
probabilities = hist.astype(float) / std_dev_image.size

optimal_alpha = recursive_otsu_thresholding(image, block_size)

if optimal_alpha is not None:
    print("Optimal alpha:", optimal_alpha)
else:
    print("Optimal alpha not found within specified iterations.")