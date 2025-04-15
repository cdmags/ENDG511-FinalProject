import tkinter as tk
from tkinter import filedialog
import torch
from ultralytics import YOLO 
from steering import GatedSteeringNet
import cv2


CLASS_MAP = {0: "blue", 1: "yellow", 2: "orange"}
COLOR_MAP = {"blue": (255, 0, 0), "yellow": (0, 255, 255), "orange": (0, 128, 255)}

# Commented on the side is the correlation to steering
FINAL_FEATURES = ['weighted_y_blue', # -0.45
                  'weighted_y_yellow', # 0.45
                  'weighted_h_blue', # -0.41
                  'weighted_h_yellow', # 0.41
                  'foreground_blue', # 0.39
                  'foreground_yellow', # -0.39
                  'diff_weighted_y', # -0.58
                  'mean_y_blue', # -0.46
                  'mean_y_yellow', # 0.46
                  'std_x_blue', # 0.54
                  'std_x_yellow', # -0.54
                  'parallax_dx_blue', # -0.55
                  'parallax_dx_yellow', # -0.55
                  'cone_angle_blue', # 0.55
                  'cone_angle_yellow', # 0.55
                  'diff_n'] # 0.57


def extract_background_aware_features(cones_by_class):

    def weighted_avg(values, weights):
        total = sum(weights)

        return sum(v*w for v, w in zip(values, weights)) / total if total > 0 else 0.0 # Calculate the weighted average using the corresponding weight

    features = {f"weighted_{metric}_{color}": weighted_avg([c[idx] for c in cones_by_class.get(color, [])], # The value is whichever feature (idx index from x, y, or h) is used for weighted average
                                                           [c[3] for c in cones_by_class.get(color, [])]) # The weight is the height (index 3)
                                                           for color in ["blue", "yellow", "orange"]
                                                           for idx, metric in enumerate(["x", "y", "h"])
                                                           if cones_by_class.get(color, []) # If the feature is distinct to the cone color, then get the weighted average 
                                                           }

    features.update({f"n_{color}": len(cones_by_class.get(color, [])) for color in ["blue", "yellow", "orange"]})
    features["diff_n"] = features.get("n_blue", 0) - features.get("n_yellow", 0) # Include the diff_in feature

    return features



def extract_feature_vector_from_yolo_output(detections, device="cuda"):

    frame = {"blue": [], "yellow": [], "orange": []}

    for cls, x, y, w, h in detections:

        # Go through every single bounding box detected in that given frame
        # Get the cone class from CLASS_MAP global var
        class_name = CLASS_MAP.get(int(cls))

        if class_name:
            frame[class_name].append((x, y, w, h))

    # Use the feature extraction function based on the each cone class, get all the features
    features = extract_background_aware_features(frame) 

    # For the final features chosen (as listed in the global var), put the features into a feature vector, else 0.0
    vector = [features.get(f, 0.0) for f in FINAL_FEATURES] 

    return torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)




def predict_steering_from_image(image, yolo_model, steering_model, device="cuda"):

    results = yolo_model(image)[0] # The model inference
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        x, y, w, h = box.xywhn[0].tolist() 
        detections.append((cls_id, x, y, w, h)) # Append the bounidng box to the detections

    input_tensor = extract_feature_vector_from_yolo_output(detections, device)

    with torch.no_grad():
        prediction = steering_model(input_tensor).item()

    return prediction, results




def process_video_with_steering(video_path, yolo_model, steering_model, device="cuda"):

    cap = cv2.VideoCapture(video_path) # Use VideoCapture for real-time video feed
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        steering, results = predict_steering_from_image(frame, yolo_model, steering_model, device) # Run the prediction on each frame using this function

        # Draw bounding boxes and confidence
        for box in results.boxes:

            cls_id = int(box.cls.item())
            class_name = CLASS_MAP.get(cls_id, "")
            color = COLOR_MAP.get(class_name, (255, 255, 255))

            conf = float(box.conf.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            label = f"{conf:.3f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


        h, w, _ = frame.shape

        steering_offset = 0 # -10.9 was used in older iterations of the steering model
        steering_scaled = (steering * 90) + steering_offset # Convert the -1 and 1 to a total of +/- 90 degrees of steering
        
        # Left or right is only triggered if the predicted steering is greater than 1 degree.
        if steering_scaled < -1:
            steering_label = "Left"

        elif steering_scaled > 1:
            steering_label = "Right"

        else:
            steering_label = "Straight"

        steering_text = f"{steering_label} [{abs(steering_scaled * 2):.1f}]"

        # Overlay steering prediction text
        text_size = cv2.getTextSize(steering_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        text_x = (w - text_size[0]) // 2
        text_y = int(h/8) + 20
        cv2.rectangle(frame, (text_x - 15, text_y - text_size[1] - 15), (text_x + text_size[0] + 15, text_y + 15), (0, 0, 0), -1)
        cv2.putText(frame, steering_text, (text_x , text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Steering Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()




root = tk.Tk()
root.withdraw()  

# Selecting files

print("Select YOLO model (.pt)")
model_path_cone = filedialog.askopenfilename(title="Select YOLO model", filetypes=[("PyTorch model", "*.pt")])
if not model_path_cone:
    raise ValueError("Model file not selected.")

print("Select steering model (.pt)")
model_path_steer = filedialog.askopenfilename(title="Select steering model", filetypes=[("PyTorch model", "*.pt")])
if not model_path_steer:
    raise ValueError("Model file not selected.")

print("Select input video (.mp4 or .avi)")
video_path = filedialog.askopenfilename(title="Select input video", filetypes=[("Video files", "*.mp4 *.avi")])
if not video_path:
    raise ValueError("Video file not selected.")

yolo_model = YOLO(model_path_cone).cuda()

steering_model = torch.load(model_path_steer, map_location = "cuda")
steering_model.eval()

process_video_with_steering(video_path, yolo_model, steering_model, device="cuda") # Main video function
