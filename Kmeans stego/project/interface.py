import tkinter as tk
import subprocess

# Color constants
DARK_GREY = '#121212'
MEDIUM_GREY = '#1F1B24'
OCEAN_BLUE = '#464EB8'
WHITE = "white"

# Font constants
BUTTON_FONT = ("Times New Roman", 15)

def run_code1():
    subprocess.Popen(
        [
            "python",
            r"Kmeans stego\project\staganography_kmeans.py",
        ]
    )


def run_code2():
    subprocess.Popen(
        [
            "python",
            r"Kmeans stego\project\Steganography.py",
        ]
    )


# Create the main window
root = tk.Tk()
root.title("Python Code Executor")

# Create a frame to contain the widgets
frame = tk.Frame(root,bg=DARK_GREY)
frame.grid(row=0, column=0, sticky=tk.NSEW)

# Create a button to run code 1
button_code1 = tk.Button(
    frame, text="K-means clustering", font=BUTTON_FONT, bg=OCEAN_BLUE, fg=WHITE, command=run_code1
)
button_code1.pack(pady=15, padx = 15)

# Create a button to run code 2
button_code2 = tk.Button(
    frame, text="Audio Video Text and Image", font=BUTTON_FONT, bg=OCEAN_BLUE, fg=WHITE, command=run_code2
)
button_code2.pack(pady=5)

# Run the Tkinter event loop
root.mainloop()
