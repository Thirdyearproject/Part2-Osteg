import tkinter as tk
import subprocess


def run_code1():
    subprocess.Popen(
        [
            "python",
            r"C:\Users\HP\Documents\GitHub\Part2-Osteg\Kmeans stego\project\staganography_kmeans.py",
        ]
    )


def run_code2():
    subprocess.Popen(
        [
            "python",
            r"C:\Users\HP\Documents\GitHub\Part2-Osteg\Kmeans stego\project\Steganography.py",
        ]
    )


# Create the main window
root = tk.Tk()
root.title("Python Code Executor")

# Create a frame to contain the widgets
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

# Create a button to run code 1
button_code1 = tk.Button(
    frame, text="steganorapy with k-means clustering", command=run_code1
)
button_code1.pack(pady=5)

# Create a button to run code 2
button_code2 = tk.Button(
    frame, text="steganorapy with audio video text and image", command=run_code2
)
button_code2.pack(pady=5)

# Run the Tkinter event loop
root.mainloop()
