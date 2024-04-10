import numpy as np
from PIL import Image
import math
import random

def calculate_std_dev(image, block_size):
    image_array = np.array(image, dtype=np.float32)
    height, width = image_array.shape[:2]
    step = block_size // 2
    std_dev_image = np.zeros((height, width), dtype=np.float32)
    for y in range(0, height - block_size + 1, step):
        for x in range(0, width - block_size + 1, step):
            block = image_array[y:y+block_size, x:x+block_size]
            std_dev = np.std(block)
            std_dev_image[y:y+block_size, x:x+block_size] = std_dev
    return std_dev_image

def otsu_threshold(std_dev_image):
    std_dev_values = std_dev_image.flatten()
    hist = np.histogram(std_dev_values, bins=256)[0]
    hist = hist.astype(float) / std_dev_values.size
    best_threshold = 0
    max_variance = 0
    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = 1 - w0
        u0 = np.sum(np.arange(t) * hist[:t]) / w0 if w0 > 0 else 0
        u1 = np.sum(np.arange(t, 256) * hist[t:]) / w1 if w1 > 0 else 0
        variance = w0 * w1 * (u0 - u1) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    xi = best_threshold
    return xi

def fake_embedding(image_array, witness_image):
    stego_array = image_array.copy()
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            if witness_image[y, x] == 255:
                if np.random.rand() < 0.5:
                    stego_array[y, x] += 1
    return stego_array

def recursive_otsu_thresholding(image, block_size, epsilon=0.1, max_iterations=10):
    xi = otsu_threshold(calculate_std_dev(image, block_size))
    alpha = 1
    for k in range(max_iterations):
        threshold = alpha * xi
        witness_cover = np.zeros_like(image, dtype=np.uint8)
        witness_stego = np.zeros_like(image, dtype=np.uint8)
        witness_cover[calculate_std_dev(image, block_size) > threshold] = 255
        stego_image_array = fake_embedding(np.array(image), witness_cover)
        witness_stego[calculate_std_dev(Image.fromarray(stego_image_array), block_size) > threshold] = 255
        if np.array_equal(witness_cover, witness_stego):
            return alpha
        alpha += k * epsilon
    return None

# Load the image
image_path = 'image.jpg'
image = Image.open(image_path)

# Split into RGB channels
red_channel = image.split()[0]
green_channel = image.split()[1]
blue_channel = image.split()[2]

# Define block size
block_size = 8

# Calculate standard deviation images for each channel
red_std_dev_image = calculate_std_dev(red_channel, block_size)
green_std_dev_image = calculate_std_dev(green_channel, block_size)
blue_std_dev_image = calculate_std_dev(blue_channel, block_size)

# Apply Otsu's method and recursive optimization for each channel
red_alpha = recursive_otsu_thresholding(red_channel, block_size)
green_alpha = recursive_otsu_thresholding(green_channel, block_size)
blue_alpha = recursive_otsu_thresholding(blue_channel, block_size)

# Print the alpha values for each channel
print("Red channel alpha:", red_alpha)
print("Green channel alpha:", green_alpha)
print("Blue channel alpha:", blue_alpha)

def t_n(x, y):
    return 0.4 - 6 / (1 + x**2 + y**2)

def ikeda_map(u, x, y):
    xn = 1 + u * (x * math.cos(t_n(x, y)) - y * math.sin(t_n(x, y)))
    yn = u * (x * math.sin(t_n(x, y)) + y * math.cos(t_n(x, y)))
    return [xn, yn]

def generate_ikeda_sequence(num_points, num_iterations, u, bound):
    l = [[random.uniform(-bound, bound), random.uniform(-bound, bound)] for _ in range(num_points)]
    for _ in range(num_iterations):
        l = [ikeda_map(u, point[0], point[1]) for point in l]
    return l

num_points = 10
num_iterations = 100
u = 0.8
bound = 5
ikeda_sequence = generate_ikeda_sequence(num_points, num_iterations, u, bound)

secret_key = (ikeda_sequence, red_alpha, green_alpha, blue_alpha)

print("Secret Key:",secret_key)

print("Alpha values (RGB):", red_alpha, green_alpha, blue_alpha)
