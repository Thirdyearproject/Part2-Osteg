import math
import binascii

# encryption_keys = []


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
def encryption():
    global string_pt
    global string_decrypt
    encryption_keys = []

    string_pt = input("\nEnter the message to be encrypted (8 Characters only): ")
    plain_text = str_to_bin(string_pt)

    string_key = input("Enter the key to be used for encryption (8 Characters only): ")
    key = str_to_bin(string_key)

    with open("key.txt", "w") as file1:
        file1.write(key)

    encryption_keys = generate_round_keys(key, encryption_keys)

    cipher_text = des_algorithm(plain_text, encryption_keys)

    with open("cipher.txt", "w") as file2:
        file2.write(cipher_text)

    print("\n---------------------Text Encrypted Successfully!---------------------\n")


# DES decryption function
def decryption():
    global string_pt
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


if __name__ == "__main__":
    encryption()
    decryption()
