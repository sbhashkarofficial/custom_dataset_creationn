import cv2
import numpy as np
import os
import json

# Global variables to track mouse state and coordinates
drawing = False      # True if mouse is pressed down and dragging
ix, iy = -1, -1      # Initial X and Y coordinates when clicked
cr_state = 0

# Define your label details
label_text = ["photo", "header", "textarea", "signature", "mrz"]
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
text_color = (0, 255, 0)       # Green text
bg_color = (255, 255, 255)     # White background for text box

# Global variables that will be modified inside the mouse callback
img = None
img_copy = None
img_states = []  # Keeps history for Undo function
cr_annotation = np.zeros((len(label_text), 2, 2), dtype=int)

def draw_rectangle_with_cursor(event, x, y, flags, param):
    global ix, iy, drawing, img, img_copy, cr_state, img_states, cr_annotation
    
    # 1. Capture initial click down (Start Point)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # 2. Capture dragging (Shows a live preview box while moving)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    # 3. Capture mouse release (End Point & Finalize Box)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
        # Prevent crashes if the user draws more boxes than available labels
        if cr_state >= len(label_text):
            print(f"All {len(label_text)} labels already assigned! Press 'q' to move to next image.")
            return

        current_label = label_text[cr_state]

        # Save a backup of the current clean state BEFORE adding the new rectangle
        img_states.append(img_copy.copy())

        # Save coordinates to annotation array BEFORE updating cr_state index position
        cr_annotation[cr_state] = [[ix, iy], [x, y]]

        # Get text width and height for background box layout
        (text_w, text_h), baseline = cv2.getTextSize(current_label, font, font_scale, font_thickness)
        
        # Draw background solid rectangle for text
        cv2.rectangle(img_copy, (ix, iy - text_h - 10), (ix + text_w, iy), bg_color, -1)
        
        # Pass the current indexed label string to the text renderer
        cv2.putText(img_copy, current_label, (ix, iy - 5), font, font_scale, text_color, font_thickness)

        # Draw the permanent bounding box outline onto img_copy
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

        img = img_copy.copy()
        cr_state += 1  # Move index forward for next box drawing action

def annotate_image(img_path):
    global img, img_copy, cr_state, img_states, cr_annotation, raw_image_shape, scale_ratio
    
    # Reset internal processing states
    cr_state = 0
    img_states = []
    cr_annotation = np.zeros((len(label_text), 2, 2), dtype=int)

    # 1. Load the raw high-resolution image
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"Warning: Could not load image file {img_path}. Skipping.")
        return False

    # 2. Calculate scaling ratio to fit the window cleanly on your monitor
    h, w, _ = raw_img.shape
    max_width = 1000  # 💡 Change this to 800 or 1200 depending on your screen resolution
    
    if w > max_width:
        scale_ratio = max_width / float(w)
        target_height = int(h * scale_ratio)
        # Resize using INTER_AREA (best interpolation method for shrinking images)
        img = cv2.resize(raw_img, (max_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        scale_ratio = 1.0
        img = raw_img.copy()

    # Create the display buffer canvas
    img_copy = img.copy()

    # 3. Create a window that locks its size to the image dimensions
    cv2.namedWindow('Draw Window', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Draw Window', draw_rectangle_with_cursor)

    print(f"\n--- Annotating: {os.path.basename(img_path)} (Scaled to {img.shape[1]}x{img.shape[0]}) ---")
    print("  - Click & Drag left mouse button to draw labeled regions.")
    print("  - Press 'z' to Undo last rectangle.")
    print("  - Press 'q' to Save and skip to NEXT image.")

    while True:
        cv2.imshow('Draw Window', img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Save & Next
            break
            
        elif key == ord('z'):  # Undo
            if cr_state > 0:
                cr_state -= 1
                img_copy = img_states.pop()
                img = img_copy.copy()
                print(f"Undo triggered. Active item reset to label: '{label_text[cr_state]}'")
            else:
                print("Nothing left to undo!")

    return True

# --- Main Runtime Processing Pipeline Loop ---
img_type = "aze_passport"
img_dir_path = "../dataset/images/" + img_type

if not os.path.exists(img_dir_path):
    print(f"Error: Target directory '{img_dir_path}' does not exist.")
    exit()


images = os.listdir(img_dir_path)
annotations = {}

print(f"{len(images)} images loaded")

for item in images:
    full_path = os.path.join(img_dir_path, item)
    
    if os.path.isfile(full_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
        success = annotate_image(full_path)
        
        if success:
            annotations_dict = {}
            for i in range(len(label_text)):
                # 4. CRUCIAL: Scale the screen coordinates BACK UP to match original raw dimensions
                scaled_coords = (cr_annotation[i] / scale_ratio).astype(int)
                coords = scaled_coords.tolist()
                annotations_dict[label_text[i]] = coords
            
            annotations[item] = annotations_dict
            print(f"Saved correctly scaled mapping structures for {item}")

    # --- DUMP DATA ---
    output_json_path = "../custom_dataset/annotations/" + img_type + ".json"
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(annotations, json_file, indent=4)

    print(f"\n All Done! File saved successfully at: {output_json_path}")

cv2.destroyAllWindows()

