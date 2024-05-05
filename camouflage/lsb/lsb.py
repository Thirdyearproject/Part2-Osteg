import os
import numpy as np
from PIL import Image

k = 3

# Global variables
binary_counter = 0
buffer_counter = 0
finalarr = bytearray((335 * 187) * 3)
clustercount = [0] * k
clustercounter = 0
cipherarr = bytearray((335 * 187) * 3)


class Pixel:
    def __init__(self, red, green, blue, cluster):
        self.red = red
        self.green = green
        self.blue = blue
        self.cluster = cluster


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

    # Now finalarr holds the LSBs of all pixels
    print("Final array:", finalarr)

    # Convert bytearray to string for further processing if needed
    finalarr_str = finalarr.decode("utf-8")
    print("Final array as string:", finalarr_str)


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
                if not line:
                    break
                r, g, b, cluster = map(int, file.readline().split())
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

    print("Final Array:")
    print(cipherarr.decode())

    with open("cipher_afterlsb.txt", "wb") as file:
        file.write(cipherarr)


def main():
    sorting_clusters()
    lsb_stegnography()
    lsb_stegnography_d()


if __name__ == "__main__":
    main()
