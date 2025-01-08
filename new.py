import tkinter as tk
from tkinter import messagebox
import subprocess
from tkinter import PhotoImage

def start_detection():
    try:
        # Run the detection script
        subprocess.run(["python", "drow.py"])
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Initialize Tkinter
root = tk.Tk()
root.title("Drowsiness Detection System")
root.geometry("500x300")
root.resizable(False, False)

# Add a background color
root.configure(bg="#f0f8ff")  # Light blue background

# Add a title label with styling
title_label = tk.Label(
    root,
    text="Drowsiness Detection System",
    font=("Arial", 20, "bold"),
    bg="#f0f8ff",
    fg="#333333"
)
title_label.pack(pady=30)

# Add a start button with enhanced styling
start_button = tk.Button(
    root,
    text="Start Detection",
    font=("Arial", 16),
    bg="#4CAF50",  # Green button
    fg="white",
    activebackground="#45a049",
    activeforeground="white",
    command=start_detection,
    width=20,
    height=2
)
start_button.pack(pady=30)

# Add an exit button
exit_button = tk.Button(
    root,
    text="Exit",
    font=("Arial", 14),
    bg="#f44336",  # Red button
    fg="white",
    activebackground="#d32f2f",
    activeforeground="white",
    command=root.quit,
    width=15,
    height=1
)
exit_button.pack(pady=10)

# Add a footer label for attribution
footer_label = tk.Label(
    root,
    text="Powered by AI Drowsiness Detector",
    font=("Arial", 10, "italic"),
    bg="#f0f8ff",
    fg="#555555"
)
footer_label.pack(side="bottom", pady=10)

# Run the Tkinter event loop
root.mainloop()
