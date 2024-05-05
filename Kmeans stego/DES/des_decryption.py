import binascii

round_keys = []
decrypted_text = ""


def bin_to_str(bin_input):
    string_output = ""
    for i in range(0, len(bin_input), 8):
        byte = bin_input[i : i + 8]
        string_output += chr(int(byte, 2))
    return string_output


def bin_to_dec(binary):
    decimal = 0
    for i in range(len(binary)):
        decimal += int(binary[len(binary) - 1 - i]) * (2**i)
    return decimal


def dec_to_bin(decimal):
    binary = bin(decimal)[2:]
    return binary.zfill(4)


def substring(input_str, position, length):
    return input_str[position : position + length]


def shift_left_one(input_str):
    return input_str[1:] + input_str[0]


def shift_left_two(input_str):
    return input_str[2:] + input_str[0] + input_str[1]


def xor(a, b):
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def generate_round_keys(key):
    # The PC1 table
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

    # The PC2 table
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

    # Compressing the key using the PC1 table
    perm_key = "".join(key[i - 1] for i in pc1)

    # Dividing the result into two equal halves
    left = perm_key[:28]
    right = perm_key[28:]

    # Generating 16 keys
    for i in range(16):
        # For rounds 1, 2, 9, 16 the key_chunks are shifted by one; for other rounds: its two left shifts
        if i in (0, 1, 8, 15):
            left_shifted = shift_left_one(left)
            right_shifted = shift_left_one(right)
        else:
            left_shifted = shift_left_two(left)
            right_shifted = shift_left_two(right)

        left = left_shifted
        right = right_shifted

        # The chunks are combined
        combined_key = left + right

        # Finally, the PC2 table is used to transpose the key bits
        round_key = "".join(combined_key[i - 1] for i in pc2)
        round_keys.append(round_key)


def des_decryption(cipher_text):
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
        perm += cipher_text[index - 1]
    

    left = perm[:32]
    right = perm[32:]

    for i in range(16):
        right_expanded = ""
        for index in expansion_table:
            right_expanded += right[index - 1]
        xored = xor(round_keys[len(round_keys)-i-1],right_expanded)
        
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
    decrypted_text = "".join(combined_text[i - 1] for i in inverse_permutation)
    return decrypted_text


def main():
    decryption_keys = [""] * 16
    with open("key.txt", "r") as file1:
        key = file1.read().strip()

    with open("cipher.txt", "r") as file2:
        cipher_text = file2.read().strip()

    generate_round_keys(key)
    decryption_keys[:] = round_keys[::-1]

    decrypted_text = des_decryption(cipher_text)

    string_decrypt = bin_to_str(decrypted_text)
    print("\nDecrypted message is:", string_decrypt)
    print("\n---------------------Text Decrypted Successfully!---------------------\n")


if __name__ == "__main__":
    main()
