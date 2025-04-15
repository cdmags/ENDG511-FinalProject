import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import shutil


root = tk.Tk()
root.withdraw()

print("Select the base image directory (containing 'train' and 'val')...")
base_image_dir = Path(filedialog.askdirectory(title="Select Base Image Directory"))
print(f"Selected base image directory: {base_image_dir}")

print("Select the base label directory (containing 'train' and 'val')...")
base_label_dir = Path(filedialog.askdirectory(title="Select Base Label Directory"))
print(f"Selected base label directory: {base_label_dir}")

print("Select the output directory for anomalies...")
output_dir = Path(filedialog.askdirectory(title="Select Output Directory"))
print(f"Selected output directory: {output_dir}")

print("Select the trained model file (.pt)...")
model_path = filedialog.askopenfilename(title="Select YOLO Model File", filetypes=[("PyTorch Model", "*.pt")])
print(f"Selected model: {model_path}")

conf_thresh = 0.2
ious_thresh = 0.18  # consider as false positive if IoU < this

output_dir.mkdir(parents=True, exist_ok=True)
model = YOLO(model_path)

# Get screen resolution and set half window size
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = screen_width // 2
window_height = screen_height // 2


def compute_iou(box1, box2):
    """Computes the IOU (intersection over union) value between predicted bounding box and the ground truth bounding box.

    Args:
        box1 (list): List of predicted bounding box parameters of cone class, x and y centers, height, and width.
        box2 (list): List of ground truth bounding box parameters of cone class, x and y centers, height, and width.

    Returns:
        float: iou_value
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou_value = inter_area / union_area if union_area > 0 else 0

    return iou_value

# === Main Loop ===
for sub_dir in ['train', 'val']:
    image_dir = base_image_dir / sub_dir
    label_dir = base_label_dir / sub_dir

    print(f"Processing {sub_dir} set from {image_dir}")

    for img_path in tqdm(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))):
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        label_file = label_dir / (img_path.stem + ".txt")
        box_groundtruth = []
        existing_labels = []

        if label_file.exists():
            with open(label_file, 'r') as f:

                for line in f.readlines():
                    parts = line.strip().split()

                    if len(parts) >= 5:
                        cls_id, xc, yc, bw, bh = map(float, parts[:5])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        box_groundtruth.append([x1, y1, x2, y2])
                        existing_labels.append([cls_id, xc, yc, bw, bh])

        # Inference
        prediction = model(img_path, verbose=False)[0].boxes

        for i, box in enumerate(prediction.xyxy.cpu().numpy()):
            conf = prediction.conf[i].item()
            cls = int(prediction.cls[i].item())

            if conf < conf_thresh:
                continue

            pred_box = box.astype(int).tolist()
            max_iou = max([compute_iou(pred_box, gt) for gt in box_groundtruth], default=0)

            if max_iou < ious_thresh:
                x1, y1, x2, y2 = np.clip(pred_box, 0, [w, h, w, h])
                crop = image[y1:y2, x1:x2]

                if crop.size > 0:
                    annotated = "Yes" if max_iou >= 0.3 else "No"
                    print(f"Prediction from {img_path.name} - Annotated: {annotated}, IoU: {max_iou:.2f}, Confidence: {conf:.2f}, Class: {cls}")

                    display_img = crop.copy()

                    scale_factor = min(window_width / display_img.shape[1], window_height / display_img.shape[0])
                    if scale_factor < 1 or display_img.shape[0] < window_height or display_img.shape[1] < window_width:
                        display_img = cv2.resize(display_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

                    original_with_box = image.copy()
                    # Draw existing GT boxes in white
                    for gt_box in box_groundtruth:
                        gx1, gy1, gx2, gy2 = gt_box
                        cv2.rectangle(original_with_box, (gx1, gy1), (gx2, gy2), (255, 255, 255), 1)

                    # Draw current predicted anomaly in green
                    iou_text = f"IoU: {max_iou:.2f}"
                    cv2.putText(original_with_box, iou_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(original_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  
                    max_display_width, max_display_height = 960, 540
                    scale = min(max_display_width / w, max_display_height / h, 1.0)
                    resized_original = cv2.resize(original_with_box, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                    cv2.imshow("Original Image with Box", resized_original)

                    window_name = f"Crop Review - Annotated: {annotated}, Model class: {cls}, Conf: {conf:.2f}"

                    cv2.imshow(window_name, display_img)
                    print("Press '0' for blue, '1' for yellow, '2' for orange, 'k' to skip (not a cone), 's' to hard skip, or 'q' to quit")
                    
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()                 

                    if key == ord('s'):
                        continue

                    if key == ord('q'):
                        print("Exiting...")
                        exit()

                    if key in [ord('0'), ord('1'), ord('2')]:                        
                        cone_id = int(chr(key))

                        # Convert bounding box to YOLO format
                        xc = ((x1 + x2) / 2) / w
                        yc = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h

                        # Check if similar label already exists using IoU
                        proposed_box = [int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)]
                        already_exists = any(compute_iou(proposed_box, [int((lbl[1] - lbl[3] / 2) * w), int((lbl[2] - lbl[4] / 2) * h), int((lbl[1] + lbl[3] / 2) * w), int((lbl[2] + lbl[4] / 2) * h)]) > 0.9 for lbl in existing_labels)

                        if not already_exists:
                            new_annotation = f"\n{cone_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                            with open(label_file, 'a') as f:
                                f.write('' + new_annotation if not new_annotation.startswith('') else new_annotation)

                            # Also save to a separate anomaly label file and image for backup
                            anomaly_txt_path = output_dir / "anom" / "ref"/ f"{img_path.stem}_anomaly_{i}_reference.txt"
                            anomaly_img_path = output_dir / "anom" / f"{img_path.stem}_anomaly_{i}.jpg"

                            # Ensure minimum crop dimensions of 12 pixels
                            crop_h, crop_w = crop.shape[:2]
                            if crop_h < 10 or crop_w < 10:
                                scale = 12 / min(crop_h, crop_w)
                                crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_CUBIC)

                            # Save original bounding box annotation
                            with open(anomaly_txt_path, 'w') as f:
                                f.write(new_annotation)

                            # Save full-cone annotation for cropped training
                            cropped_train_txt = output_dir / "anom" / f"{img_path.stem}_anomaly_{i}.txt"
                            with open(cropped_train_txt, 'w') as f:
                                f.write(f"{cone_id} 0.500000 0.500000 1.000000 1.000000")

                            cv2.imwrite(str(anomaly_img_path), crop)

                    elif key == ord('k'):
                        filename = f"{img_path.stem}_fp{i}"
                        img_out_path = output_dir / "fp" / f"{filename}.jpg"
                        txt_out_path = output_dir / "fp" / f"{filename}.txt"

                        # Ensure minimum crop dimensions of 12 pixels
                        crop_h, crop_w = crop.shape[:2]
                        if crop_h < 10 or crop_w < 10:
                            scale = 12 / min(crop_h, crop_w)
                            crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_CUBIC)

                        cv2.imwrite(str(img_out_path), crop)
                        with open(txt_out_path, 'w') as f:
                            pass  # write empty label file

print("Hard negatives and relabeled cones have been added to the original annotation files.")
