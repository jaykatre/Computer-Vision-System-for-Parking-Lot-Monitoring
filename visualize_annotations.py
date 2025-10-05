import cv2
import os
import random

# --- Configuration ---
image_dir = "train/images"
label_dir = "train/labels"
output_image_path = "annotated_image.jpg" # Where to save the output

# Class names mapping (Assuming based on web_app.py)
class_names = {
    0: 'space-empty',
    1: 'space-occupied'
}

# Colors for bounding boxes (BGR format)
colors = {
    'space-empty': (0, 255, 0),    # Green
    'space-occupied': (0, 0, 255) # Red
}
default_color = (255, 255, 255) # White for unknown classes
# --- End Configuration ---

def draw_annotations(image_path, label_path, output_path):
    """Reads an image and its YOLO annotations, draws the boxes, and saves."""
    
    # Check if image and label files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        # Continue without labels if desired, or return
        print("Proceeding without annotations for this image.")
        # return # Uncomment to stop if labels are missing

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return
        
    h, w, _ = image.shape
    print(f"Processing image: {os.path.basename(image_path)} (Dimensions: {w}x{h})")

    # Read and draw annotations if label file exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Skipping invalid line in {label_path}: {line.strip()}")
                        continue
                        
                    class_id = int(parts[0])
                    x_center, y_center, box_w, box_h = map(float, parts[1:])

                    # Denormalize coordinates
                    x_center *= w
                    y_center *= h
                    box_w *= w
                    box_h *= h

                    # Calculate top-left (x1, y1) and bottom-right (x2, y2)
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    # Get class name and color
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    color = colors.get(class_name, default_color)

                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Put label text above the box
                    label = f"{class_name}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10 + label_height # Adjust position if box is near top
                    cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                except ValueError as e:
                    print(f"Warning: Skipping malformed line in {label_path}: {line.strip()} - Error: {e}")
                except Exception as e:
                     print(f"An unexpected error occurred processing line: {line.strip()} in {label_path} - Error: {e}")


    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")

    # Optionally display the image
    # cv2.imshow("Annotated Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Looking for images and labels...")
    
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' not found.")
        exit()
    if not os.path.isdir(label_dir):
        print(f"Error: Label directory '{label_dir}' not found.")
        exit()

    try:
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        print(f"Error listing files in image directory '{image_dir}': {e}")
        exit()

    if not image_files:
        print(f"No image files found in {image_dir}")
    else:
        # Select the first image found
        selected_image_name = image_files[500]
        image_path = os.path.join(image_dir, selected_image_name)
        
        # Construct the corresponding label file path
        base_name = os.path.splitext(selected_image_name)[0]
        label_filename = f"{base_name}.txt"
        label_path = os.path.join(label_dir, label_filename)

        draw_annotations(image_path, label_path, output_image_path) 