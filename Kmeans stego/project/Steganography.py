import numpy as np
import pandas as pand
import os
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog


def txt_encode(text, stxt):
    l = len(text)
    i = 0
    add = ""
    while i < l:
        t = ord(text[i])
        if t >= 32 and t <= 64:
            t1 = t + 48
            t2 = t1 ^ 170  # 170: 10101010
            res = bin(t2)[2:].zfill(8)
            add += "0011" + res

        else:
            t1 = t - 48
            t2 = t1 ^ 170
            res = bin(t2)[2:].zfill(8)
            add += "0110" + res
        i += 1
    res1 = add + "111111111111"
    print(
        "The string after binary conversion appyling all the transformation :- "
        + (res1)
    )
    length = len(res1)
    print("Length of binary after conversion:- ", length)
    HM_SK = ""
    ZWC = {"00": "\u200C", "01": "\u202C", "11": "\u202D", "10": "\u200E"}
    file1 = open("Sample_cover_files/cover_text.txt", "r+")
    nameoffile = stxt
    file3 = open(nameoffile, "w+", encoding="utf-8")
    word = []
    for line in file1:
        word += line.split()
    i = 0
    while i < len(res1):
        s = word[int(i / 12)]
        j = 0
        x = ""
        HM_SK = ""
        while j < 12:
            x = res1[j + i] + res1[i + j + 1]
            HM_SK += ZWC[x]
            j += 2
        s1 = s + HM_SK
        file3.write(s1)
        file3.write(" ")
        i += 12
    t = int(len(res1) / 12)
    while t < len(word):
        file3.write(word[t])
        file3.write(" ")
        t += 1
    file3.close()
    file1.close()
    print("\nStego file has successfully generated")


def encode_txt_data(text1, stxt, count_label):
    count2 = 0
    file1 = open("Sample_cover_files/cover_text.txt", "r")
    for line in file1:
        for word in line.split():
            count2 = count2 + 1
    file1.close()
    bt = int(count2)
    max_words_message = "Maximum number of words that can be inserted: {}".format(
        int(bt / 6)
    )
    count_label.config(text=max_words_message)
    l = len(text1)
    if l <= bt:
        print("\nInputed message can be hidden in the cover file\n")
        txt_encode(text1, stxt)
    else:
        print("\nString is too big please reduce string size")
        encode_txt_data()


def BinaryToDecimal(binary):
    string = int(binary, 2)
    return string


def decode_txt_data(stego):
    ZWC_reverse = {"\u200C": "00", "\u202C": "01", "\u202D": "11", "\u200E": "10"}
    file4 = open(stego, "r", encoding="utf-8")
    temp = ""
    for line in file4:
        for words in line.split():
            T1 = words
            binary_extract = ""
            for letter in T1:
                if letter in ZWC_reverse:
                    binary_extract += ZWC_reverse[letter]
            if binary_extract == "111111111111":
                break
            else:
                temp += binary_extract
    print("\nEncrypted message presented in code bits:", temp)
    lengthd = len(temp)
    print("\nLength of encoded bits:- ", lengthd)
    i = 0
    a = 0
    b = 4
    c = 4
    d = 12
    final = ""
    while i < len(temp):
        t3 = temp[a:b]
        a += 12
        b += 12
        i += 12
        t4 = temp[c:d]
        c += 12
        d += 12
        if t3 == "0110":
            decimal_data = BinaryToDecimal(t4)
            final += chr((decimal_data ^ 170) + 48)
        elif t3 == "0011":
            decimal_data = BinaryToDecimal(t4)
            final += chr((decimal_data ^ 170) - 48)
    return final


def msgtobinary(msg):
    if type(msg) == str:
        result = "".join([format(ord(i), "08b") for i in msg])

    elif type(msg) == bytes or type(msg) == np.ndarray:
        result = [format(i, "08b") for i in msg]

    elif type(msg) == int or type(msg) == np.uint8:
        result = format(msg, "08b")

    else:
        raise TypeError("Input type is not supported in this function")

    return result


def encode_img_data(img, data, nameoffile):
    if len(data) == 0:
        raise ValueError("Data entered to be encoded is empty")

    no_of_bytes = (img.shape[0] * img.shape[1] * 3) // 8

    print("\t\nMaximum bytes to encode in Image :", no_of_bytes)

    if len(data) > no_of_bytes:
        raise ValueError(
            "Insufficient bytes Error, Need Bigger Image or give Less Data !!"
        )

    data += "*^*^*"

    binary_data = msgtobinary(data)
    print("\n")
    print(binary_data)
    length_data = len(binary_data)

    print("\nThe Length of Binary data", length_data)

    index_data = 0

    for i in img:
        for pixel in i:
            r, g, b = msgtobinary(pixel)
            if index_data < length_data:
                pixel[0] = int(r[:-1] + binary_data[index_data], 2)
                index_data += 1
            if index_data < length_data:
                pixel[1] = int(g[:-1] + binary_data[index_data], 2)
                index_data += 1
            if index_data < length_data:
                pixel[2] = int(b[:-1] + binary_data[index_data], 2)
                index_data += 1
            if index_data >= length_data:
                break
    cv2.imwrite(nameoffile, img)
    print(
        "\nEncoded the data successfully in the Image and the image is successfully saved with name ",
        nameoffile,
    )


def decode_img_data(img):
    data_binary = ""
    for i in img:
        for pixel in i:
            r, g, b = msgtobinary(pixel)
            data_binary += r[-1]
            data_binary += g[-1]
            data_binary += b[-1]
            total_bytes = [
                data_binary[i : i + 8] for i in range(0, len(data_binary), 8)
            ]
            decoded_data = ""
            for byte in total_bytes:
                decoded_data += chr(int(byte, 2))
                if decoded_data[-5:] == "*^*^*":
                    return decoded_data[:-5]


def encode_aud_data(data, stegofile):
    import wave

    song = wave.open("Sample_cover_files/cover_audio.wav", mode="rb")

    nframes = song.getnframes()
    frames = song.readframes(nframes)
    frame_list = list(frames)
    frame_bytes = bytearray(frame_list)

    res = "".join(format(i, "08b") for i in bytearray(data, encoding="utf-8"))
    print("\nThe string after binary conversion :- " + (res))
    length = len(res)
    print("\nLength of binary after conversion :- ", length)

    data = data + "*^*^*"

    result = []
    for c in data:
        bits = bin(ord(c))[2:].zfill(8)
        result.extend([int(b) for b in bits])

    j = 0
    for i in range(0, len(result), 1):
        res = bin(frame_bytes[j])[2:].zfill(8)
        if res[len(res) - 4] == result[i]:
            frame_bytes[j] = frame_bytes[j] & 253  # 253: 11111101
        else:
            frame_bytes[j] = (frame_bytes[j] & 253) | 2
            frame_bytes[j] = (frame_bytes[j] & 254) | result[i]
        j = j + 1

    frame_modified = bytes(frame_bytes)

    with wave.open(stegofile, "wb") as fd:
        fd.setparams(song.getparams())
        fd.writeframes(frame_modified)
    print("\nEncoded the data successfully in the audio file.")
    song.close()


def decode_aud_data(nameoffile):
    import wave

    song = wave.open(nameoffile, mode="rb")

    nframes = song.getnframes()
    frames = song.readframes(nframes)
    frame_list = list(frames)
    frame_bytes = bytearray(frame_list)

    extracted = ""
    p = 0
    for i in range(len(frame_bytes)):
        if p == 1:
            break
        res = bin(frame_bytes[i])[2:].zfill(8)
        if res[len(res) - 2] == 0:
            extracted += res[len(res) - 4]
        else:
            extracted += res[len(res) - 1]

        all_bytes = [extracted[i : i + 8] for i in range(0, len(extracted), 8)]
        decoded_data = ""
        for byte in all_bytes:
            decoded_data += chr(int(byte, 2))
            if decoded_data[-5:] == "*^*^*":
                p = 1
                return decoded_data[:-5]


def KSA(key):
    key_length = len(key)
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % key_length]) % 256
        S[i], S[j] = S[j], S[i]
    return S


def PRGA(S, n):
    i = 0
    j = 0
    key = []
    while n > 0:
        n = n - 1
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        key.append(K)
    return key


def preparing_key_array(s):
    return [ord(c) for c in s]


def encryption(plaintext):
    print("Enter the key : ")
    key = input()
    key = preparing_key_array(key)

    S = KSA(key)

    keystream = np.array(PRGA(S, len(plaintext)))
    plaintext = np.array([ord(i) for i in plaintext])

    cipher = keystream ^ plaintext
    ctext = ""
    for c in cipher:
        ctext = ctext + chr(c)
    return ctext


def decryption(ciphertext):
    print("Enter the key : ")
    key = input()
    key = preparing_key_array(key)

    S = KSA(key)

    keystream = np.array(PRGA(S, len(ciphertext)))
    ciphertext = np.array([ord(i) for i in ciphertext])

    decoded = keystream ^ ciphertext
    dtext = ""
    for c in decoded:
        dtext = dtext + chr(c)
    return dtext


def embed(frame):
    data = input("\nEnter the data to be Encoded in Video :")
    data = encryption(data)
    print("The encrypted data is : ", data)
    if len(data) == 0:
        raise ValueError("Data entered to be encoded is empty")

    data += "*^*^*"

    binary_data = msgtobinary(data)
    length_data = len(binary_data)

    index_data = 0

    for i in frame:
        for pixel in i:
            r, g, b = msgtobinary(pixel)
            if index_data < length_data:
                pixel[0] = int(r[:-1] + binary_data[index_data], 2)
                index_data += 1
            if index_data < length_data:
                pixel[1] = int(g[:-1] + binary_data[index_data], 2)
                index_data += 1
            if index_data < length_data:
                pixel[2] = int(b[:-1] + binary_data[index_data], 2)
                index_data += 1
            if index_data >= length_data:
                break
        return frame


def extract(frame):
    data_binary = ""
    final_decoded_msg = ""
    for i in frame:
        for pixel in i:
            r, g, b = msgtobinary(pixel)
            data_binary += r[-1]
            data_binary += g[-1]
            data_binary += b[-1]
            total_bytes = [
                data_binary[i : i + 8] for i in range(0, len(data_binary), 8)
            ]
            decoded_data = ""
            for byte in total_bytes:
                decoded_data += chr(int(byte, 2))
                if decoded_data[-5:] == "*^*^*":
                    for i in range(0, len(decoded_data) - 5):
                        final_decoded_msg += decoded_data[i]
                    final_decoded_msg = decryption(final_decoded_msg)
                    print(
                        "\n\nThe Encoded data which was hidden in the Video was :--\n",
                        final_decoded_msg,
                    )
                    return


def encode_vid_data(n):
    cap = cv2.VideoCapture("Sample_cover_files/cover_video.mp4")
    vidcap = cv2.VideoCapture("Sample_cover_files/cover_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    size = (frame_width, frame_height)
    out = cv2.VideoWriter("stego_video.mp4", fourcc, 25.0, size)
    max_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        max_frame += 1
    cap.release()
    print("Total number of Frame in selected Video :", max_frame)

    frame_number = 0
    frame_ = None  # Initialize frame_ outside the loop
    while vidcap.isOpened():
        frame_number += 1
        ret, frame = vidcap.read()
        if ret == False:
            break
        if frame_number == n:
            change_frame_with = embed(frame)
            frame_ = change_frame_with
            frame = change_frame_with
    out.write(frame)

    print("\nEncoded the data successfully in the video file.")
    return frame_


def decode_vid_data(frame_, n):
    cap = cv2.VideoCapture("stego_video.mp4")

    max_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        max_frame += 1
    print("Total number of Frame in selected Video :", max_frame)

    vidcap = cv2.VideoCapture("stego_video.mp4")
    frame_number = 0
    while vidcap.isOpened():
        frame_number += 1
        ret, frame = vidcap.read()
        if ret == False:
            break
        if frame_number == n:
            return extract(frame_)


class TextStegWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Text Steganography Operations")
        self.geometry("400x300")

        self.create_widgets()

    def create_widgets(self):
        self.encode_button = tk.Button(
            self, text="Encode Text", command=self.encode_text
        )
        self.encode_button.pack()

        self.decode_button = tk.Button(
            self, text="Decode Text", command=self.decode_text
        )
        self.decode_button.pack()

        self.exit_button = tk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=20)

    def encode_text(self):
        encrypt_window = tk.Toplevel(self)
        encrypt_window.title("Encrypt Text")

        encrypt_label = tk.Label(encrypt_window, text="Enter text to encrypt:")
        encrypt_label.pack()

        encrypt_entry = tk.Entry(encrypt_window)
        encrypt_entry.pack()

        text_label = tk.Label(
            encrypt_window,
            text="Enter the name of the Stego file after Encoding(with extension):",
        )
        text_label.pack()

        text_entry = tk.Entry(encrypt_window)
        text_entry.pack()

        count_label = tk.Label(encrypt_window, text="")
        count_label.pack()

        def perform_txt_encription():
            message = encrypt_entry.get()
            txt = text_entry.get()
            try:
                encode_txt_data(message, txt, count_label)
                # If encryption is successful, update the success message
                success_label.config(text="Text successfully encoded!")
            except Exception as e:
                # If encryption fails, update the error message
                success_label.config(text=f"Encryption failed: {str(e)}")

        encrypt_button = tk.Button(
            encrypt_window, text="Encrypt", command=perform_txt_encription
        )
        encrypt_button.pack()

        success_label = tk.Label(encrypt_window, text="", fg="green")
        success_label.pack()

    def decode_text(self):
        decrypt_window = tk.Toplevel(self)
        decrypt_window.title("Decrypt Text")

        decrypt_label = tk.Label(
            decrypt_window,
            text="Enter the stego file name(with extension) to decode the message:",
        )
        decrypt_label.pack()

        decrypt_entry = tk.Entry(decrypt_window)
        decrypt_entry.pack()

        def perform_txt_decryption():
            stego_file = decrypt_entry.get()
            try:
                decoded_message = decode_txt_data(stego_file)
                # Display the decoded message
                decoded_message_label.config(text="Decoded Message: " + decoded_message)
            except Exception as e:
                # If decoding fails, display the error message
                decoded_message_label.config(text="Decoding failed: " + str(e))

        decrypt_button = tk.Button(
            decrypt_window, text="Decrypt", command=perform_txt_decryption
        )
        decrypt_button.pack()

        decoded_message_label = tk.Label(decrypt_window, text="", fg="blue")
        decoded_message_label.pack()


class ImageStegWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Image Steganography Operations")
        self.geometry("400x300")

        self.create_widgets()

    def create_widgets(self):
        self.encode_button = tk.Button(
            self, text="Encode Image", command=self.encode_image
        )
        self.encode_button.pack()

        self.decode_button = tk.Button(
            self, text="Decode Image", command=self.decode_image
        )
        self.decode_button.pack()

        self.exit_button = tk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=20)

    def encode_image(self):
        encode_window = tk.Toplevel(self)
        encode_window.title("Encode Image")

        data_label = tk.Label(encode_window, text="Enter data to encode:")
        data_label.pack()

        data_entry = tk.Entry(encode_window)
        data_entry.pack()

        file_label = tk.Label(
            encode_window,
            text="Enter the name of the Stego image file after encoding (with extension):",
        )
        file_label.pack()

        file_entry = tk.Entry(encode_window)
        file_entry.pack()

        try:

            def perform_image_encoding():
                data = data_entry.get()
                image = cv2.imread("Sample_cover_files/cover_image.jpg")
                file_name = file_entry.get()
                # Call the encode_img_data function passing data and file_name
                encode_img_data(image, data, file_name)
                success_label.config(text="Image successfully encoded!")

            encode_button = tk.Button(
                encode_window, text="Encode", command=perform_image_encoding
            )
            encode_button.pack()

        except Exception as e:
            success_label.config(text=f"Encoding failed: {str(e)}")

        success_label = tk.Label(encode_window, text="", fg="green")
        success_label.pack()

    def decode_image(self):
        decode_window = tk.Toplevel(self)
        decode_window.title("Decode Image")

        stego_label = tk.Label(
            decode_window,
            text="Enter the stego image file name (with extension) to decode the message:",
        )
        stego_label.pack()

        stego_entry = tk.Entry(decode_window)
        stego_entry.pack()

        def perform_image_decoding():
            stego_file = stego_entry.get()
            image1 = cv2.imread(stego_file)
            try:
                decoded_message = decode_img_data(image1)
                decoded_message_label.config(text="Decoded Message: " + decoded_message)
            except Exception as e:
                decoded_message_label.config(text="Decoding failed: " + str(e))

        decode_button = tk.Button(
            decode_window, text="Decode", command=perform_image_decoding
        )
        decode_button.pack()

        decoded_message_label = tk.Label(decode_window, text="", fg="blue")
        decoded_message_label.pack()


class AudioStegWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Audio Steganography Operations")
        self.geometry("400x300")

        self.create_widgets()

    def create_widgets(self):
        self.encode_button = tk.Button(
            self, text="Encode Audio", command=self.encode_audio
        )
        self.encode_button.pack()

        self.decode_button = tk.Button(
            self, text="Decode Audio", command=self.decode_audio
        )
        self.decode_button.pack()

        self.exit_button = tk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=20)

    def encode_audio(self):
        encode_window = tk.Toplevel(self)
        encode_window.title("Encode Audio")

        # file_label = tk.Label(
        #     encode_window,
        #     text="Enter the name of the  audio file to encoding (with extension):",
        # )
        # file_label.pack()

        # file_entry = tk.Entry(encode_window)
        # file_entry.pack()

        data_label = tk.Label(encode_window, text="Enter data to encode:")
        data_label.pack()

        data_entry = tk.Entry(encode_window)
        data_entry.pack()

        stego_label = tk.Label(
            encode_window,
            text="Enter the name of the Stego audio file after encoding (with extension):",
        )
        stego_label.pack()

        stego_entry = tk.Entry(encode_window)
        stego_entry.pack()

        try:

            def perform_audio_encoding():
                data = data_entry.get()
                # file_name = file_entry.get()
                stego_name = stego_entry.get()

                # Call the encode_aud_data function passing data and file_name
                encode_aud_data(data, stego_name)
                success_label.config(text="Audio successfully encoded!")

            encode_button = tk.Button(
                encode_window, text="Encode", command=perform_audio_encoding
            )
            encode_button.pack()

        except Exception as e:
            success_label.config(text=f"Encoding failed: {str(e)}")

        success_label = tk.Label(encode_window, text="", fg="green")
        success_label.pack()

    def decode_audio(self):
        decode_window = tk.Toplevel(self)
        decode_window.title("Decode Audio")

        stego_label = tk.Label(
            decode_window,
            text="Enter the stego audio file name (with extension) to decode the message:",
        )
        stego_label.pack()

        stego_entry = tk.Entry(decode_window)
        stego_entry.pack()

        def perform_audio_decoding():
            stego_file = stego_entry.get()
            try:
                decoded_message = decode_aud_data(stego_file)
                decoded_message_label.config(text="Decoded Message: " + decoded_message)
            except Exception as e:
                decoded_message_label.config(text="Decoding failed: " + str(e))

        decode_button = tk.Button(
            decode_window, text="Decode", command=perform_audio_decoding
        )
        decode_button.pack()

        decoded_message_label = tk.Label(decode_window, text="", fg="blue")
        decoded_message_label.pack()


class VideoStegWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Video Steganography Operations")
        self.geometry("400x300")
        self.secret = None
        self.create_widgets()

    def create_widgets(self):
        self.encode_button = tk.Button(
            self, text="Encode Video", command=self.encode_video
        )
        self.encode_button.pack()

        self.decode_button = tk.Button(
            self, text="Decode Video", command=self.decode_video
        )
        self.decode_button.pack()

        self.exit_button = tk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=20)

    def encode_video(self):
        encode_window = tk.Toplevel(self)
        encode_window.title("Encode Video")

        data_label = tk.Label(
            encode_window, text="Enter the frame number where you want to embed data : "
        )
        data_label.pack()

        data_entry = tk.Entry(encode_window)
        data_entry.pack()

        try:

            def perform_video_encoding():
                data = data_entry.get()
                # Call the encode_vid_data function passing data and file_name
                self.secret = encode_vid_data(data)
                success_label.config(text="Video successfully encoded!")

            encode_button = tk.Button(
                encode_window, text="Encode", command=perform_video_encoding
            )
            encode_button.pack()

        except Exception as e:
            success_label.config(text=f"Encoding failed: {str(e)}")

        success_label = tk.Label(encode_window, text="", fg="green")
        success_label.pack()

    def decode_video(self):
        decode_window = tk.Toplevel(self)
        decode_window.title("Decode Video")

        stego_label = tk.Label(
            decode_window,
            text="Enter the secret frame number from where you want to extract data",
        )
        stego_label.pack()

        stego_entry = tk.Entry(decode_window)
        stego_entry.pack()

        def perform_video_decoding():
            stego_file = stego_entry.get()
            try:
                decoded_message = decode_vid_data(self.secret, stego_file)
                decoded_message_label.config(
                    text="Decoded Message: " + str(decoded_message)
                )
            except Exception as e:
                decoded_message_label.config(text="Decoding failed: " + str(e))

        decode_button = tk.Button(
            decode_window, text="Decode", command=perform_video_decoding
        )
        decode_button.pack()

        decoded_message_label = tk.Label(decode_window, text="", fg="blue")
        decoded_message_label.pack()


class SteganographyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Steganography App")
        self.geometry("400x300")

        self.create_widgets()

    def create_widgets(self):
        # Main Menu
        self.main_menu_label = tk.Label(self, text="Main Menu", font=("Arial", 14))
        self.main_menu_label.pack(pady=10)

        self.text_steg_button = tk.Button(
            self, text="Text Steganography", command=self.text_steg
        )
        self.text_steg_button.pack()

        self.image_steg_button = tk.Button(
            self, text="Image Steganography", command=self.image_steg
        )
        self.image_steg_button.pack()

        self.audio_steg_button = tk.Button(
            self, text="Audio Steganography", command=self.audio_steg
        )
        self.audio_steg_button.pack()

        self.video_steg_button = tk.Button(
            self, text="Video Steganography", command=self.video_steg
        )
        self.video_steg_button.pack()

        self.exit_button = tk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=20)

    def text_steg(self):
        text_steg_window = TextStegWindow(self)
        text_steg_window.grab_set()

    def image_steg(self):
        image_steg_window = ImageStegWindow(self)
        image_steg_window.grab_set()

    def audio_steg(self):
        audio_steg_window = AudioStegWindow(self)
        audio_steg_window.grab_set()

    def video_steg(self):
        video_steg_window = VideoStegWindow(self)
        video_steg_window.grab_set()


def main():

    app = SteganographyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
