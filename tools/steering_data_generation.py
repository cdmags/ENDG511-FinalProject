import cv2
import numpy as np
import mss
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import ctypes
import mmap
import struct
import os

# Open GUI to select YOLO model
root = tk.Tk()
root.withdraw()

print("Select YOLO model (.pt)")
model_path = filedialog.askopenfilename(title="Select YOLO model", filetypes=[("PyTorch model", "*.pt")])
if not model_path:
    raise ValueError("Model file not selected.")

# Load YOLO model
model = YOLO(model_path)

# Define colors & class names
class_colors = {0: (255, 0, 0), 1: (0, 255, 255), 2: (0, 165, 255)}
class_names = {0: "blue", 1: "yellow", 2: "orange"}

# Setting screen capture to full screeen
sct = mss.mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

width = monitor["width"]
height = monitor["height"]
fps = 20

# Setting up video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated_screen_output.mp4', fourcc, fps, (width, height))

# Creating folders if they don't exist to save cone data and driving telemetry later
os.makedirs("frame_yolo_labels", exist_ok=True)
os.makedirs("frame_driving_data", exist_ok=True)

# Assetto Corsa shared memory layout
class SPageFilePhysics(ctypes.Structure):
    _fields_ = [
        ('packetId', ctypes.c_int),
        ('gas', ctypes.c_float),
        ('brake', ctypes.c_float),
        ('fuel', ctypes.c_float),
        ('gear', ctypes.c_int),
        ('rpm', ctypes.c_int),
        ('steerAngle', ctypes.c_float),
        ('speedKmh', ctypes.c_float),
    ]

physics_size = ctypes.sizeof(SPageFilePhysics)

# Open shared memory
physics = mmap.mmap(-1, physics_size, "Local\\acpmf_physics")

# Setting frame counter to 0. This is just used for file naming
frame_count = 0

# Main loop
while True:
    # Take a screenshot of the monitor and convert it to a numpy array and BGR color format so that it works with OpenCV
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame)[0]

    # Creating the file name based on the frame count
    yolo_filename = f"frame_yolo_labels/frame_{frame_count:04d}.txt"
    driving_data_filename = f"frame_driving_data/frame_{frame_count:04d}.txt"

    # Using cone detection model to predict cone locations and then saving them into the yolo labels folder in the yolo format
    with open(yolo_filename, mode='w') as yolo_file:

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = class_colors.get(cls_id, (255, 255, 255))
            label = f"{conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            bx, by, bw, bh = box.xywhn[0].tolist()
            
            yolo_file.write(f"{int(box.cls)} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    # Displaying the screenshot with the cones with the predicted boxes on my second monitor for visualization
    out.write(frame)
    cv2.imshow("Live Screen Detection", frame)

    # Retreive Assetto Corsa Telemetry
    physics.seek(0)
    buf = physics.read(physics_size)

    data = SPageFilePhysics.from_buffer_copy(buf)

    # print(
    #     f"Speed: {data.speedKmh:.1f} km/h | "
    #     f"Steering: {data.steerAngle*90:.1f}° | "
    #     f"Throttle: {data.gas:.2f} | "
    #     f"Brake: {data.brake:.2f} | "
    #     f"RPM: {data.rpm} | "
    #     f"Gear: {data.gear}"
    # )
    

    # Write Assetto Corsa telemetry into a new file
    with open(driving_data_filename, mode='w') as driving_data_file:
        driving_data_file.write(f"{data.steerAngle*90:.3f} {data.gas:.3f} {data.brake:.3f} {data.gear} {data.speedKmh:.2f} {data.rpm}\n")

    # Increase frame count for file labelling
    frame_count += 1

    # Pressing 'q' on keyboards exits the while loop and ends the programs
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closing necessary things
out.release()
cv2.destroyAllWindows()
