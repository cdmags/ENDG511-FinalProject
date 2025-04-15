import os
import cv2
import glob
from pathlib import Path

# Modify to the correct image and label paths
image_dir = Path("Final Project/dataset_yolo_mod/images/train")        
label_dir = Path("Final Project/dataset_yolo_mod/labels/train")           
output_dir = None   
show_images = True                   

# Cones class colors, note that cv2 uses BGR format
class_colors = {0: (255, 0, 0), 1: (0, 255, 255), 2: (0, 165, 255)}

def resize_to_fit_window(image, max_width=1600, max_height=900):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))

# draw function to draw the bounding boxes
def draw_yolo_boxes(image, label_path):
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_c, y_c, bw, bh = map(float, parts)
            cls_id = int(cls_id)

            x_c *= w
            y_c *= h
            bw *= w
            bh *= h

            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            color = class_colors.get(cls_id, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image

def draw_new_boxes(image, class_id):
    boxes = []
    drawing = False
    ix, iy = -1, -1

    draw_color = class_colors[class_id]

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, image, boxes
        temp = image.copy()

        if event == cv2.EVENT_LBUTTONDOWN: # use the mouse buttons to draw the boxes and update the frame as it is drawn
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.rectangle(temp, (ix, iy), (x, y), draw_color, 2)
            cv2.imshow("Draw Bounding Box", temp)
            cv2.moveWindow("Draw Bounding Box", 20, 20)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(temp, (ix, iy), (x, y), draw_color, 2)

            box_drawn = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
            boxes.append(box_drawn)

            print(f"New box drawn: {box_drawn}")
            cv2.imshow("Draw Bounding Box", temp)
            cv2.moveWindow("Draw Bounding Box", 20, 20)

            image = temp.copy() # Show the bounding box just drawn to be able to compare

    cv2.namedWindow("Draw Bounding Box")
    cv2.setMouseCallback("Draw Bounding Box", draw_rectangle)
    cv2.imshow("Draw Bounding Box", image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('y'): 
            return image, boxes

        elif cv2.waitKey(1) == 27: # Esc
            break

    return None, None


image_paths = glob.glob(os.path.join(image_dir, '*.*'))

total = len(image_paths)
index = 0

# Conitnuously loop to update the cv2 frame depending on the keys pressed
while 0 <= index < total:

    image_path = image_paths[index]
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, base + '.txt')

    if not os.path.exists(label_path):
        print(f"Label file not found for image: {image_path}")
        index += 1
        continue

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if image is None:
        print(f"Could not load image: {image_path}")
        index += 1
        continue
    
    # Use a copy of the frame image when drawing the YOLO boxes
    annotated = draw_yolo_boxes(image.copy(), label_path) # Save it as the "annotated" image

    max_display_width, max_display_height = 1600, 900
    scale = min(max_display_width / w, max_display_height / h, 1.0)

    if show_images:

        # Resize with the annotated image, don't have to recalculate bounding boxes
        display_image = resize_to_fit_window(annotated, max_display_width, max_display_height) 

        cv2.imshow("Ground Truth", display_image) # The rescaled image is what gets displayed
        cv2.moveWindow("Ground Truth", 20, 20) 

        print(f"[{index+1}/{total}] Showing: {base}")
        key = cv2.waitKeyEx(0)

        if key == 27:  # Esc
            break

        elif key == ord('j'):  # Jump to filename if you want to skip to a specific image
            filename = input("Enter filename to jump to (without extension): ").strip()
            matches = [i for i, p in enumerate(image_paths) if os.path.splitext(os.path.basename(p))[0] == filename]
            if matches:
                index = matches[0]

            else:
                print("File not found.")
                continue

        elif key == 2424832:  # Left arrow 
            index = max(0, index - 1)

        elif key == 2555904:  # Right arrow
            index = min(total - 1, index + 1)

        elif key == ord('d'):
            drawing = True

            while drawing:
                print("Press '0', '1', or '2' to select class and draw.")
                key_draw = cv2.waitKey(0)
                cv2.destroyAllWindows()
                class_id = int(chr(key_draw))

                draw_img = display_image.copy() # Make a copy of the annotated display image so you see boxes as you draw

                drawn_img, drawn_boxes = draw_new_boxes(draw_img, class_id)

                with open(label_path, 'a') as f:
                    print("Drawing complete, writing to file...")

                    if drawn_boxes is not None:

                        # Convert the x and y coordinates of the drawn boxes to YOLO format and scale based on resized window
                        for x1, y1, x2, y2 in drawn_boxes:
                            xc = ((x1 + x2) / 2) / int(w * scale)
                            yc = ((y1 + y2) / 2) / int(h * scale)
                            bw = (x2 - x1) / int(w * scale)
                            bh = (y2 - y1) / int(h * scale)
                            
                            f.write(f"\n{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                
                label_path = os.path.join(label_dir, base + '.txt')

                # Show the most up to date image after drawing by reloading the label paths
                if drawn_img is not None:
                    drawn_annotated = draw_yolo_boxes(drawn_img.copy(), label_path) 
                else:
                    drawn_annotated = draw_yolo_boxes(draw_img.copy(), label_path)

                drawn_display_image = resize_to_fit_window(drawn_annotated, max_display_width, max_display_height)

                cv2.imshow(f"Draw Bounding Box", drawn_display_image) # Display the most up to date image
                cv2.moveWindow("Draw Bounding Box", 20, 20)

                print("Draw again? Press 'y' / 'n' to confirm.") # Ask to redraw and loop again
                key_draw_end = cv2.waitKey(0)
                if key_draw_end == ord('y'):
                    continue

                elif key_draw_end == ord('n'):
                    drawing = False
                    cv2.destroyAllWindows()

        else:
            index += 1

    if output_dir:
        output_path = os.path.join(output_dir, base + '_gt.jpg')
        cv2.imwrite(output_path, annotated)

cv2.destroyAllWindows()
