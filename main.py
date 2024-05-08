import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import math

# Color constants
DARK_GREY = '#121212'
MEDIUM_GREY = '#1F1B24'
OCEAN_BLUE = '#464EB8'
WHITE = "white"

# Font constants
FONT = ("Times New Roman", 17)
BUTTON_FONT = ("Times New Roman", 15)

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

def ikeda_map_modified(x_n, x_0, mu=1, h=0.5, m=20):
    return x_n + h * (-mu * x_n + m * math.sin(x_0))

def generate_ikeda_sequence(x0, sequence_length, mu=1, h=0.5, m=20):
    sequence = [x0]  # Initialize sequence with initial condition
    x_n = x0  # Initial value of x_n
    for _ in range(sequence_length - 1):  # Generate sequence up to sequence_length - 1
        x_n = ikeda_map_modified(x_n, sequence[-1], mu, h, m)  # Update x_n using the last value in sequence
        sequence.append(x_n)  # Append updated x_n to sequence
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
    delimiter = "000000"
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
        for channel in range(1):
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

def extract_secret_bits(image_array, candidate_blocks, x0, alpha_values):
    block_size = 4  # Example block size (adjust as needed)
    alpha_R, alpha_G, alpha_B = alpha_values

    extracted_bits = []

    for idx, (y, x) in enumerate(candidate_blocks):
        keystream = generate_keystream(block_size, x0)
        offsets = [(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 1)]
        selected_offsets = [offsets[i] for i in range(len(offsets)) if keystream[i] == 1]
        neighbor_pixels = get_neighbor_pixels(image_array, y, x, selected_offsets)

        extracted_bit_sequence = ""

        for channel in range(1):
            central_pixel_binary = format(image_array[y, x, channel], '08b')
            neighbor_pixels_binary = ''.join(format(p[channel], '08b') for p in neighbor_pixels)
            binary_sequence = neighbor_pixels_binary + central_pixel_binary

            num_padding_bits = 128 - len(binary_sequence)
            padding_sequence = generate_ikeda_sequence(x0, num_padding_bits)
            padding_bits = ''.join(str(int(x * 1000) % 2) for x in padding_sequence)

            final_sequence = binary_sequence + padding_bits
            parity = sum(int(bit) for bit in final_sequence) % 2

            # Extracted bit is the parity bit of the final sequence
            extracted_bit = parity

            extracted_bit_sequence += str(extracted_bit)

        extracted_bits.append(extracted_bit_sequence)

    return extracted_bits

def decrypt_secret_message(extracted_bits):
    # Remove delimiter from extracted bits
    delimiter = "000000"
    extracted_bits = ''.join(extracted_bits).split(delimiter)[0]

    # Convert binary string to characters
    message = ""
    for i in range(0, len(extracted_bits), 8):
        byte = extracted_bits[i:i+8]
        message += chr(int(byte, 2))

    return message

def embed_message():
    image_path = filedialog.askopenfilename(title="Select Image File")
    if not image_path:
        return
    
    x0 = float(x0_entry.get())
    secret_message = secret_message_entry.get()
    secret_message = secret_message + 'a'
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Additional parameters for embedding
    block_size = 8  # Adjust as needed

    # Calculate standard deviation images and alpha values (using recursive Otsu's method)
    red_std_dev_image = calculate_std_dev(image.split()[0], block_size)
    green_std_dev_image = calculate_std_dev(image.split()[1], block_size)
    blue_std_dev_image = calculate_std_dev(image.split()[2], block_size)
    red_alpha = recursive_otsu_thresholding(image.split()[0], block_size)
    green_alpha = recursive_otsu_thresholding(image.split()[1], block_size)
    blue_alpha = recursive_otsu_thresholding(image.split()[2], block_size)

    # Get candidate blocks
    candidate_blocks = get_candidate_blocks(image_array, (red_alpha, green_alpha, blue_alpha))

    # Prepare secret message bits
    secret_bits = [int(bit) for bit in ''.join(format(ord(c), '08b') for c in secret_message)]

    # Effective embedding
    effective_embedding(image_array, candidate_blocks, secret_bits, x0, (red_alpha, green_alpha, blue_alpha))

    # Save the stego image
    stego_image_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Save Stego Image As")
    if not stego_image_path:
        return
    stego_image = Image.fromarray(image_array)
    stego_image.save(stego_image_path)

    result_label.config(text="Embedding completed. Stego image saved as " + stego_image_path)

def decrypt_message():
    stego_image_path = filedialog.askopenfilename(title="Select Stego Image File")
    if not stego_image_path:
        return
    
    x0 = float(x0_entry.get())

    # Load stego image
    stego_image = Image.open(stego_image_path)
    stego_image_array = np.array(stego_image)

    # Additional parameters for decryption
    block_size = 8  # Adjust as needed

    # Calculate standard deviation images and alpha values for the stego image
    red_std_dev_image_stego = calculate_std_dev(stego_image.split()[0], block_size)
    green_std_dev_image_stego = calculate_std_dev(stego_image.split()[1], block_size)
    blue_std_dev_image_stego = calculate_std_dev(stego_image.split()[2], block_size)
    red_alpha_stego = recursive_otsu_thresholding(stego_image.split()[0], block_size)
    green_alpha_stego = recursive_otsu_thresholding(stego_image.split()[1], block_size)
    blue_alpha_stego = recursive_otsu_thresholding(stego_image.split()[2], block_size)

    # Get candidate blocks from the stego image
    candidate_blocks_stego = get_candidate_blocks(stego_image_array, (red_alpha_stego, green_alpha_stego, blue_alpha_stego))

    # Extract secret bits from the stego image
    extracted_bits = extract_secret_bits(stego_image_array, candidate_blocks_stego, x0, (red_alpha_stego, green_alpha_stego, blue_alpha_stego))

    # Decrypt the secret message from the extracted bits
    decrypted_message = decrypt_secret_message(extracted_bits)

    result_label.config(text="Decrypted Message: " + decrypted_message[:-1])

# Create GUI
root = tk.Tk()
root.title("Steganography Tool")
root.configure(bg=DARK_GREY)

# Embedding Frame
embed_frame = tk.LabelFrame(root,font=FONT, bg=MEDIUM_GREY, fg=WHITE, text="Embed Message")
embed_frame.pack(padx=10, pady=10, fill="both", expand=True)

# x0 Entry
x0_label = tk.Label(embed_frame,font=FONT, bg=MEDIUM_GREY, fg=WHITE, text="x0:")
x0_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
x0_entry = tk.Entry(embed_frame)
x0_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# Secret Message Entry
secret_message_label = tk.Label(embed_frame, font=FONT, bg=MEDIUM_GREY, fg=WHITE, text="Secret Message:")
secret_message_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
secret_message_entry = tk.Entry(embed_frame)
secret_message_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

# Embed Button
embed_button = tk.Button(embed_frame,font=BUTTON_FONT, bg=OCEAN_BLUE, fg=WHITE, text="Embed", command=embed_message)
embed_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="we")

# Decryption Frame
decrypt_frame = tk.LabelFrame(root,font=FONT, bg=MEDIUM_GREY, fg=WHITE, text="Decrypt Message")
decrypt_frame.pack(padx=10, pady=10, fill="both", expand=True)

# Decryption Button
decrypt_button = tk.Button(decrypt_frame,font=BUTTON_FONT, bg=OCEAN_BLUE, fg=WHITE, text="Decrypt", command=decrypt_message)
decrypt_button.pack(padx=5, pady=5)

# Result Label
result_label = tk.Label(root, font=FONT, bg=MEDIUM_GREY, fg=WHITE,text="")
result_label.pack(padx=10, pady=5)

root.mainloop()
