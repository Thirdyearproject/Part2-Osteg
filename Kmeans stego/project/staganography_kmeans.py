import math
import random
import string
import numpy as np
from sklearn.cluster import KMeans
import imageio.v3 as imageio
import os
import numpy as np
from PIL import Image
from tkinter import filedialog

k = 3

# Global variables
binary_counter = 0
buffer_counter = 0
finalarr = bytearray((335 * 187) * 3)
clustercount = [0] * k
clustercounter = 0
cipherarr = bytearray((335 * 187) * 3)
string_pt = ""

# Constants
MAX = 100000
MAXK = 20

import tkinter as tk
from tkinter import messagebox


# Function to convert alphabetical string to binary string
def str_to_bin(string_input):
    bin_output = ""
    for char in string_input:
        bin_output += format(ord(char), "08b")
    return bin_output


# Function to convert binary string to alphabetical string
def bin_to_str(bin_input):
    string_output = ""
    for i in range(0, len(bin_input), 8):
        string_output += chr(int(bin_input[i : i + 8], 2))
    return string_output


# Function to convert binary string to decimal
def bin_to_dec(binary):
    decimal = 0
    for i in range(len(binary)):
        if binary[i] == "1":
            decimal += 2 ** (len(binary) - 1 - i)
    return decimal


# Function to convert decimal to binary string
def dec_to_bin(decimal):
    binary = format(decimal, "04b")
    return binary


def shift_left_one(input_str):
    return input_str[1:] + input_str[0]


def shift_left_two(input_str):
    return input_str[2:] + input_str[0] + input_str[1]


# Function to perform XOR operation on two strings
def xor(a, b):
    result = ""
    for i in range(len(a)):
        if a[i] != b[i]:
            result += "1"
        else:
            result += "0"
    return result


# Function to generate 16 round keys and store them in a 2D list
def generate_round_keys(key, round_key_array):
    pc1 = [
        57,
        49,
        41,
        33,
        25,
        17,
        9,
        1,
        58,
        50,
        42,
        34,
        26,
        18,
        10,
        2,
        59,
        51,
        43,
        35,
        27,
        19,
        11,
        3,
        60,
        52,
        44,
        36,
        63,
        55,
        47,
        39,
        31,
        23,
        15,
        7,
        62,
        54,
        46,
        38,
        30,
        22,
        14,
        6,
        61,
        53,
        45,
        37,
        29,
        21,
        13,
        5,
        28,
        20,
        12,
        4,
    ]
    pc2 = [
        14,
        17,
        11,
        24,
        1,
        5,
        3,
        28,
        15,
        6,
        21,
        10,
        23,
        19,
        12,
        4,
        26,
        8,
        16,
        7,
        27,
        20,
        13,
        2,
        41,
        52,
        31,
        37,
        47,
        55,
        30,
        40,
        51,
        45,
        33,
        48,
        44,
        49,
        39,
        56,
        34,
        53,
        46,
        42,
        50,
        36,
        29,
        32,
    ]
    perm_key = "".join([key[i - 1] for i in pc1])

    left = perm_key[:28]
    right = perm_key[28:]

    for i in range(16):
        if i == 0 or i == 1 or i == 8 or i == 15:
            left = shift_left_one(left)
            right = shift_left_one(right)
        else:
            left = shift_left_two(left)
            right = shift_left_two(right)

        combined_key = left + right
        round_key = ""
        for index in pc2:
            round_key += combined_key[index - 1]
        round_key_array.append(round_key)
    return round_key_array


# Function to perform DES encryption or decryption
def des_algorithm(input_text, round_keys):
    initial_permutation = [
        58,
        50,
        42,
        34,
        26,
        18,
        10,
        2,
        60,
        52,
        44,
        36,
        28,
        20,
        12,
        4,
        62,
        54,
        46,
        38,
        30,
        22,
        14,
        6,
        64,
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        57,
        49,
        41,
        33,
        25,
        17,
        9,
        1,
        59,
        51,
        43,
        35,
        27,
        19,
        11,
        3,
        61,
        53,
        45,
        37,
        29,
        21,
        13,
        5,
        63,
        55,
        47,
        39,
        31,
        23,
        15,
        7,
    ]
    expansion_table = [
        32,
        1,
        2,
        3,
        4,
        5,
        4,
        5,
        6,
        7,
        8,
        9,
        8,
        9,
        10,
        11,
        12,
        13,
        12,
        13,
        14,
        15,
        16,
        17,
        16,
        17,
        18,
        19,
        20,
        21,
        20,
        21,
        22,
        23,
        24,
        25,
        24,
        25,
        26,
        27,
        28,
        29,
        28,
        29,
        30,
        31,
        32,
        1,
    ]
    substition_boxes = [
        [
            [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
        ],
        [
            [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
        ],
        [
            [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
        ],
        [
            [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
        ],
        [
            [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
        ],
        [
            [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
        ],
        [
            [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
        ],
        [
            [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
        ],
    ]
    permutation_tab = [
        16,
        7,
        20,
        21,
        29,
        12,
        28,
        17,
        1,
        15,
        23,
        26,
        5,
        18,
        31,
        10,
        2,
        8,
        24,
        14,
        32,
        27,
        3,
        9,
        19,
        13,
        30,
        6,
        22,
        11,
        4,
        25,
    ]
    inverse_permutation = [
        40,
        8,
        48,
        16,
        56,
        24,
        64,
        32,
        39,
        7,
        47,
        15,
        55,
        23,
        63,
        31,
        38,
        6,
        46,
        14,
        54,
        22,
        62,
        30,
        37,
        5,
        45,
        13,
        53,
        21,
        61,
        29,
        36,
        4,
        44,
        12,
        52,
        20,
        60,
        28,
        35,
        3,
        43,
        11,
        51,
        19,
        59,
        27,
        34,
        2,
        42,
        10,
        50,
        18,
        58,
        26,
        33,
        1,
        41,
        9,
        49,
        17,
        57,
        25,
    ]
    perm = ""
    for index in initial_permutation:
        perm += input_text[index - 1]
    left = perm[:32]
    right = perm[32:]
    for i in range(16):
        right_expanded = ""
        for index in expansion_table:
            right_expanded += right[index - 1]

        xored = xor(round_keys[i], right_expanded)
        res = ""
        for j in range(8):
            row = bin_to_dec(xored[j * 6] + xored[j * 6 + 5])
            col = bin_to_dec(xored[j * 6 + 1 : j * 6 + 5])
            val = substition_boxes[j][row][col]
            res += dec_to_bin(val)
        perm2 = ""
        for index in permutation_tab:
            perm2 += res[index - 1]
        xored1 = xor(perm2, left)
        left = xored1
        if i < 15:
            left, right = right, xored1

    combined_text = left + right

    result_text = "".join([combined_text[i - 1] for i in inverse_permutation])
    return result_text


# DES encryption function
def encryption(string_pt, string_key):
    global string_decrypt
    encryption_keys = []

    plain_text = str_to_bin(string_pt)

    key = str_to_bin(string_key)

    with open("key.txt", "w") as file1:
        file1.write(key)

    encryption_keys = generate_round_keys(key, encryption_keys)

    cipher_text = des_algorithm(plain_text, encryption_keys)

    with open("cipher.txt", "w") as file2:
        file2.write(cipher_text)

    print("\n---------------------Text Encrypted Successfully!---------------------\n")


# DES decryption function
def decryption(string_pt):
    global string_decrypt
    initial_keys = []
    with open("key.txt", "r") as file3:
        key = file3.read()

    with open("cipher.txt", "r") as file4:
        cipher_text = file4.read()
    initial_keys = generate_round_keys(key, initial_keys)
    decryption_keys = initial_keys[::-1]
    decrypted_text = des_algorithm(cipher_text, decryption_keys)
    string_decrypt = bin_to_str(decrypted_text)
    print("Decrypted message is:", string_decrypt)

    if string_pt == string_decrypt:
        print(
            "\n---------------------Text Decrypted Successfully!---------------------\n"
        )


class Pixel:
    def __init__(self, red, green, blue, cluster=-1, min_dist=float("inf")):
        self.red = red
        self.green = green
        self.blue = blue
        self.cluster = cluster
        self.min_dist = min_dist


def kmeans_clustering(pixels, k, epochs, total_pixels):
    # Initialising cluster with -1 as it does not belong to any cluster
    # Initialising min_dist with infinity as it will decrease with number of iterations
    for pixel in pixels:
        pixel.cluster = -1
        pixel.min_dist = float("inf")

    # Initialising clusters with random centroids amongst pixels
    centroids_indices = np.random.choice(total_pixels, k, replace=False)
    centroids = [pixels[i] for i in centroids_indices]

    # Calculate distance, update centroids and repeat for given epochs
    for e in range(epochs):
        # Assigning points to clusters by minimizing euclidean distance
        for i in range(total_pixels):
            for j in range(k):
                # Calculate distance of the point to the centroid
                distance = np.sqrt(
                    (pixels[i].red - centroids[j].red) ** 2
                    + (pixels[i].green - centroids[j].green) ** 2
                    + (pixels[i].blue - centroids[j].blue) ** 2
                )
                if distance < pixels[i].min_dist:
                    pixels[i].min_dist = distance
                    pixels[i].cluster = j

        # Computing new centroids (Heart of k-means i.e. updating centroids)
        # Calculating mean of points in a cluster
        num_pixels = [0.0] * k
        sum_red = [0.0] * k
        sum_green = [0.0] * k
        sum_blue = [0.0] * k

        for i in range(total_pixels):
            cluster = pixels[i].cluster
            num_pixels[cluster] += 1
            sum_red[cluster] += pixels[i].red
            sum_green[cluster] += pixels[i].green
            sum_blue[cluster] += pixels[i].blue

            # Reset distance to max
            pixels[i].min_dist = float("inf")

        # Computing the new centroids
        print(f"Epoch - {e + 1}")
        for i in range(k):
            if (
                num_pixels[i] == 0
            ):  # If no point is assigned to the cluster, assigning new random centroid
                centroids[i] = pixels[np.random.randint(total_pixels)]
            else:  # Else, calculate mean of the points in the cluster and assign them as new centroid
                centroids[i].red = sum_red[i] / num_pixels[i]
                centroids[i].green = sum_green[i] / num_pixels[i]
                centroids[i].blue = sum_blue[i] / num_pixels[i]
            print(
                f"Centroid - {i}: ({centroids[i].red}, {centroids[i].green}, {centroids[i].blue})"
            )
        print()

    # Storing Results
    cluster_count = [0] * k
    for pixel in pixels:
        cluster_count[pixel.cluster] += 1

    # Return cluster counts and pixels
    return cluster_count, pixels


def kmeans(imagename, k, epochs):
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Image to pixels
    image_path = os.path.join(current_dir, imagename)

    # Load image
    img = imageio.imread(image_path)
    height, width, _ = img.shape
    total_pixels = height * width

    # Storing pixels
    pixels = [Pixel(*img[i, j]) for i in range(height) for j in range(width)]

    # Training the model
    cluster_count, pixels = kmeans_clustering(pixels, k, epochs, total_pixels)

    # Results
    print(f"Total pixels - {total_pixels}")
    print("Pixels in each cluster:-")
    for c, count in enumerate(cluster_count):
        print(f"Cluster-{c}: {count}")

    # Write pixel information to a text file
    with open(os.path.join(current_dir, "clusters.txt"), "w") as fp:
        for i, pixel in enumerate(pixels):
            fp.write(
                f"{i + 1} {pixel.red:.0f} {pixel.green:.0f} {pixel.blue:.0f} {pixel.cluster}\n"
            )

    # Pixels to Image
    for c in range(k):
        cluster_pixels = [
            pixels[i] for i in range(total_pixels) if pixels[i].cluster == c
        ]
        cluster_length = len(cluster_pixels)
        dim = int(np.ceil(np.sqrt(cluster_length)))  # Square root rounded up
        cluster_image = np.array(
            [[pixel.red, pixel.green, pixel.blue] for pixel in cluster_pixels],
            dtype=np.uint8,
        )
        cluster_image = np.pad(
            cluster_image, ((0, dim**2 - cluster_length), (0, 0)), mode="constant"
        )
        cluster_image = cluster_image.reshape(dim, dim, 3)
        imageio.imwrite(os.path.join(current_dir, f"cluster-{c}.png"), cluster_image)


def sorting_clusters():
    total_pixels = 335 * 187
    clustercount = [0] * k

    for i in range(k):
        with open("clusters.txt", "r") as file:
            lines = file.readlines()

        filename = f"{i}.txt"
        with open(filename, "w") as f:
            for line in lines:
                r, g, b, cluster = map(int, line.split()[1:5])
                if cluster == i:
                    f.write(f"{r} {g} {b} {cluster}\n")
                    clustercount[i] += 1

    print("\nSuccessfully sorted all the clusters\n")
    return clustercount


def lsb_insert(filename, binary_array, frame):
    # Opening each cluster file to hide data.
    with open(filename, "r") as file:
        total_pixels = sum(1 for _ in file)

    print(f"Total pixels in {filename} is {total_pixels}")

    clustercount.append(total_pixels)

    # Convert bytearray to string
    binary_string = binary_array.decode("utf-8")

    # Creating flagged binary array
    len_array = frame * 8
    flagged_binary_array = binary_string[:len_array] + "0" * 8

    print(f"Flagged binary array for file {filename} is\n{flagged_binary_array}\n")

    # LSB process
    pixel_counter = 0
    with open(filename, "r") as file:
        with open("clusters_lsb.txt", "a+") as fl:
            for j in range(len(flagged_binary_array)):
                line = file.readline()
                if not line:
                    break
                r, g, b, cluster = map(int, line.split())
                if flagged_binary_array[j % len(flagged_binary_array)] == "0":
                    r = r if r % 2 == 0 else r - 1
                elif flagged_binary_array[j % len(flagged_binary_array)] == "1":
                    r = r if r % 2 == 1 else r + 1
                j += 1
                if flagged_binary_array[j % len(flagged_binary_array)] == "0":
                    g = g if g % 2 == 0 else g - 1
                elif flagged_binary_array[j % len(flagged_binary_array)] == "1":
                    g = g if g % 2 == 1 else g + 1
                j += 1
                if flagged_binary_array[j % len(flagged_binary_array)] == "0":
                    b = b if b % 2 == 0 else b - 1
                elif flagged_binary_array[j % len(flagged_binary_array)] == "1":
                    b = b if b % 2 == 1 else b + 1
                fl.write(f"{r} {g} {b} {cluster}\n")
                pixel_counter += 1

            for _ in range(total_pixels - pixel_counter):
                line = file.readline()
                if not line:
                    break
                fl.write(line)


def lsb_fetch(filename, total_pixels):
    finalarr = bytearray()  # Resize bytearray to hold the values
    buffer_counter = 0
    with open(filename, "r") as file:
        for _ in range(total_pixels):
            line = file.readline().strip()
            if not line:
                break  # End of file or empty line, exit loop
            values = line.split()[1:5]
            if len(values) != 4:
                # print("Error: Invalid line in", filename)
                continue  # Skip invalid line
            r, g, b, _ = map(int, values)
            lsb_r = r % 2
            finalarr.append(lsb_r)
            buffer_counter += 1

            lsb_g = g % 2
            finalarr.append(lsb_g)
            buffer_counter += 1

            lsb_b = b % 2
            finalarr.append(lsb_b)
            buffer_counter += 1


def lsb_stegnography():
    binary_array = bytearray(
        b"0010111110110101111110011101110100000011111001101100011100000100"
    )
    print("Binary array:")
    print(binary_array.decode())

    frame_size = 8
    noofframes = len(binary_array) // frame_size
    framepercluster = noofframes // k
    reminder = noofframes % k

    print("No of frames:", noofframes)

    print("\nRunning LSB\n")
    for i in range(k):
        filename = f"{i}.txt"
        if i < k - 1:
            lsb_insert(filename, binary_array, framepercluster)
        else:
            lsb_insert(filename, binary_array, framepercluster + reminder)

    width, height = 335, 187
    for i in range(k):
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        with open(f"clusters_lsb.txt", "r") as file:
            for j in range(height * width):
                line = file.readline()
                if len(line) == 0:
                    continue
                if not line:
                    break
                r, g, b, cluster = map(int, line.split())
                if cluster == i:
                    pixels[j // width, j % width] = [r, g, b]

        img = Image.fromarray(pixels)
        img.save(f"cluster-{i}.png")
        print(f"cluster-{i}.png saved in same directory\n")


def lsb_stegnography_d():
    global buffer_counter, cipherarr
    k = 3
    for i in range(k):
        name = f"cluster-{i}.png"
        if not os.path.exists(name):
            print(f"Error: {name} not found")
            continue

        img = Image.open(name)
        pixels = np.array(img)
        height, width, _ = pixels.shape

        with open(f"cluster-{i}.txt", "w") as fp:
            for j in range(height):
                for k in range(width):
                    fp.write(f"{pixels[j, k, 0]} {pixels[j, k, 1]} {pixels[j, k, 2]}\n")

        lsb_fetch(f"cluster-{i}.txt", width * height)

    with open("cipher_afterlsb.txt", "wb") as file:
        file.write(cipherarr)


class Application(tk.Tk):
    global string_pt

    def __init__(self):
        super().__init__()

        self.title("Encryption/Decryption")
        self.user_choice = tk.IntVar(value=0)  # Variable to store user choice

        self.create_widgets()

    def create_widgets(self):
        # Label for displaying "STEGANOGRAPHY"
        self.steganography_label = tk.Label(
            self, text="STEGANOGRAPHY", font=("Helvetica", 16, "bold")
        )
        self.steganography_label.pack()

        # Button to encrypt
        self.encrypt_button = tk.Button(
            self, text="Encrypt", command=self.create_encrypt_window
        )
        self.encrypt_button.pack(pady=10)

        # Button to decrypt
        self.decrypt_button = tk.Button(
            self, text="Decrypt", command=self.create_decrypt_window
        )
        self.decrypt_button.pack(pady=10)

    def create_encrypt_window(self):
        encrypt_window = tk.Toplevel(self)
        encrypt_window.title("Encryption")
        encrypt_window.geometry("400x250")

        # Label and entry widget for inputting the message
        message_label = tk.Label(
            encrypt_window,
            text="Enter the message to be encrypted (8 Characters only):",
        )
        message_label.grid(row=0, column=0, padx=10, pady=5)
        message_entry = tk.Entry(encrypt_window)
        message_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and entry widget for inputting the key
        key_label = tk.Label(
            encrypt_window,
            text="Enter the key to be used for encryption (8 Characters only):",
        )
        key_label.grid(row=1, column=0, padx=10, pady=5)
        key_entry = tk.Entry(encrypt_window)
        key_entry.grid(row=1, column=1, padx=10, pady=5)

        # Button to choose the image file
        def choose_image():
            imagename = filedialog.askopenfilename()
            if imagename:
                image_entry.delete(0, tk.END)
                image_entry.insert(0, os.path.basename(imagename))

        choose_image_button = tk.Button(
            encrypt_window, text="Choose Image", command=choose_image
        )
        choose_image_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        # Entry widget for displaying the chosen image name
        image_entry = tk.Entry(encrypt_window)
        image_entry.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        # Label and entry widget for inputting the number of clusters
        clusters_label = tk.Label(
            encrypt_window,
            text="Enter the number of clusters (k):",
        )
        clusters_label.grid(row=4, column=0, padx=10, pady=5)
        clusters_entry = tk.Entry(encrypt_window)
        clusters_entry.grid(row=4, column=1, padx=10, pady=5)

        # Label and entry widget for inputting the number of epochs
        epochs_label = tk.Label(
            encrypt_window,
            text="Enter the number of iterations (epochs):",
        )
        epochs_label.grid(row=5, column=0, padx=10, pady=5)
        epochs_entry = tk.Entry(encrypt_window)
        epochs_entry.grid(row=5, column=1, padx=10, pady=5)

        # Function to perform encryption
        def perform_encryption():
            string_pt = message_entry.get()
            string_key = key_entry.get()
            imagename = image_entry.get()
            k = int(clusters_entry.get())
            epochs = int(epochs_entry.get())

            # Check if all necessary fields are filled
            if len(string_pt) != 8 or len(string_key) != 8 or not imagename:
                # Display an error message if any field is empty or the input message or key length is not 8 characters
                messagebox.showerror("Error", "Please fill all fields correctly.")
            else:
                # Call the encryption function with the input message, key, image name, number of clusters, and epochs
                encryption(string_pt, string_key)
                kmeans(imagename, k, epochs)
                sorting_clusters()
                lsb_stegnography()

        # Button to trigger encryption
        encrypt_button = tk.Button(
            encrypt_window, text="Encrypt", command=perform_encryption
        )
        encrypt_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def create_decrypt_window(self):
        decrypt_window = tk.Toplevel(self)
        decrypt_window.title("Decryption")
        decrypt_window.geometry("300x200")
        lsb_stegnography_d()
        decryption(string_pt)

        # Add widgets to the decryption window as needed


def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
