import numpy as np
from PIL import Image
import math

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

def t_n(x, y):
    return 0.4 - 6 / (1 + x**2 + y**2)

def ikeda_map_modified(x_n, x_0, mu=1, h=0.5, m=20):
    return x_n + h * (-mu * x_n + m * math.sin(x_0))

def generate_ikeda_sequence(x0, sequence_length, mu=1, h=0.5, m=20):
    sequence = []
    x_n = x0
    for _ in range(sequence_length):
        x_n = ikeda_map_modified(x_n, x0, mu, h, m)
        sequence.append(x_n)
    return sequence

def generate_keystream(block_size, x0, mu=1, h=0.5, m=20):
    sequence_length = block_size * block_size - 1
    sequence = generate_ikeda_sequence(x0, sequence_length, mu, h, m)
    keystream = [int(x * 1000) % 2 for x in sequence] 
    return keystream 
def get_neighbor_pixels(image_array, y, x, offsets):
    neighbors = []
    for dy, dx in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < image_array.shape[0] and 0 <= nx < image_array.shape[1]:
            neighbors.append(image_array[ny, nx])
    return neighbors


def effective_embedding(image_array, candidate_blocks, secret_bits, x0, alpha_values):
    block_size = 4  # Example block size (adjust as needed)
    alpha_R, alpha_G, alpha_B = alpha_values

    # Add delimiter (example: "001100")
    delimiter = "001100"
    for bit in delimiter:
        secret_bits.append(int(bit))

    # Pad secret message if necessary 
    while len(secret_bits) < len(candidate_blocks):
        secret_bits.append(0)

    for idx, (y, x) in enumerate(candidate_blocks):
        keystream = generate_keystream(block_size, x0) 
        offsets = [(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 1)]
        selected_offsets = [offsets[i] for i in range(len(offsets)) if keystream[i] == 1]
        neighbor_pixels = get_neighbor_pixels(image_array, y, x, selected_offsets)

        # Embedding in all channels (R, G, B)
        for channel in range(3):
            central_pixel_binary = format(image_array[y, x, channel], '08b')
            neighbor_pixels_binary = ''.join(format(p[channel], '08b') for p in neighbor_pixels)
            binary_sequence = neighbor_pixels_binary + central_pixel_binary

            num_padding_bits = 128 - len(binary_sequence)
            padding_sequence = generate_ikeda_sequence(x0, num_padding_bits)
            padding_bits = ''.join(str(int(x * 1000) % 2) for x in padding_sequence)

            final_sequence = binary_sequence + padding_bits
            parity = sum(int(bit) for bit in final_sequence) % 2

            secret_bit = secret_bits[idx]

            if (parity == 0 and secret_bit == 0) or (parity == 1 and secret_bit == 1):
                pass 
            else:
                image_array[y, x, channel] ^= 1
                
def get_candidate_blocks(image_array, alpha_values, block_size=8, threshold_factor=0.8):
    candidate_blocks = []
    for y in range(0, image_array.shape[0] - block_size + 1, block_size):
        for x in range(0, image_array.shape[1] - block_size + 1, block_size):
            block = image_array[y:y+block_size, x:x+block_size]
            std_devs = [np.std(block[:, :, c]) for c in range(3)]
            if all(std_dev > alpha * threshold_factor for std_dev, alpha in zip(std_devs, alpha_values)):
                candidate_blocks.append((y, x))
    return candidate_blocks

def extract_secret_bits(stego_image_array, candidate_blocks, x0, block_size=4):
    extracted_bits = []
    delimiter = "001100" 

    for y, x in candidate_blocks:
        keystream = generate_keystream(block_size, x0)
        offsets = [(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 1)]
        selected_offsets = [offsets[i] for i in range(len(offsets)) if keystream[i] == 1]
        neighbor_pixels = get_neighbor_pixels(stego_image_array, y, x, selected_offsets)

        # Extract from all channels and combine using XOR
        binary_sequence = ""
        for channel in range(3):
            central_pixel_binary = format(stego_image_array[y, x, channel], '08b')
            neighbor_pixels_binary = ''.join(format(p[channel], '08b') for p in neighbor_pixels)
            channel_sequence = neighbor_pixels_binary + central_pixel_binary
            binary_sequence += format(int(channel_sequence, 2) ^ int(binary_sequence, 2), f'0{len(channel_sequence)}b') 

        # Extract bits until delimiter is found
        for i in range(0, len(binary_sequence), len(delimiter)):
            extracted_bit_sequence = binary_sequence[i:i+len(delimiter)]
            if extracted_bit_sequence == delimiter:
                return extracted_bits  # Stop extraction after delimiter is found
            else:
                extracted_bits.append(int(extracted_bit_sequence, 2)) 

    return extracted_bits  

# --- Main Embedding Process ---

# Load the image
image_path = 'image.jpg'  # Replace with your image path
image = Image.open(image_path)
image_array = np.array(image)

# Define block size
block_size = 8

# Calculate standard deviation images and alpha values (using recursive Otsu's method)
red_std_dev_image = calculate_std_dev(image.split()[0], block_size)
green_std_dev_image = calculate_std_dev(image.split()[1], block_size)
blue_std_dev_image = calculate_std_dev(image.split()[2], block_size)
red_alpha = recursive_otsu_thresholding(image.split()[0], block_size)
green_alpha = recursive_otsu_thresholding(image.split()[1], block_size)
blue_alpha = recursive_otsu_thresholding(image.split()[2], block_size)

# Get candidate blocks
candidate_blocks = get_candidate_blocks(image_array, (red_alpha, green_alpha, blue_alpha))

# --- Secret Key and Message Preparation ---

# Secret key (replace with actual values)
secret_key = (0.1,red_alpha,green_alpha,blue_alpha)  # (x0, factor_R, factor_G, factor_B)
x0, factor_R, factor_G, factor_B = secret_key

# Calculate alpha values 
alpha_R = 1 + (factor_R * 0.1)
alpha_G = 1 + (factor_G * 0.1)
alpha_B = 1 + (factor_B * 0.1)

# Prepare secret message bits
secret_message = "Hello, world!"  # Example message 
secret_bits = [int(bit) for bit in ''.join(format(ord(c), '08b') for c in secret_message)]

# --- Perform Embedding --- 
effective_embedding(image_array, candidate_blocks, secret_bits, x0, (alpha_R, alpha_G, alpha_B))

# Save the stego image
print(secret_key)
stego_image = Image.fromarray(image_array)
stego_image.save("stego_image.png")