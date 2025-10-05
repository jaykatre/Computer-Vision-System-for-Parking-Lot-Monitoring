import cv2
from ultralytics import YOLO
import time

# --- Configuration ---
# Path to your trained model weights. Change this if necessary.
# Common location: runs/detect/train/weights/best.pt
# For Jetson Nano/TensorRT optimized model, use: runs/detect/train/weights/best.engine
model_path = "runs/detect/train9/weights/best.pt" 
# model_path = "runs/detect/train/weights/best.engine" # Uncomment this for TensorRT model

# Input source: 0 for webcam, or path to video/image file
# source = "path/to/your/video.mp4"
# source = "path/to/your/image.jpg"
source = "hello.jpg" 

# Confidence threshold for detections
conf_threshold = 0.3 
# --- End Configuration ---

# Load the trained YOLOv8 model
try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model file exists at: {model_path}")
    exit()

# Check if the source is an image or video/webcam
is_image = isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

if is_image:
    print(f"Running inference on image: {source}")
    # Run inference on the image
    results = model.predict(source=source, conf=conf_threshold, stream=False) # stream=False for single image

    # Assuming single image result
    res = results[0] 
    frame = res.orig_img # Get the original image

    # Draw bounding boxes
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{model.names[cls]}: {conf:.2f}"
        
        # Get color based on class (example: green for empty, red for occupied)
        color = (0, 255, 0) if model.names[cls] == 'space-empty' else (0, 0, 255) 
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    cv2.imshow("Parking Detection", frame)
    print("Press any key to exit.")
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

else:
    print(f"Opening video source: {source}")
    # Open the video source (webcam or file)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    prev_time = 0
    
    while True:
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            print("End of video stream or error reading frame.")
            break # Exit loop if video ends or error occurs

        # Calculate FPS
        current_time = time.time()
        fps_calc = 1 / (current_time - prev_time)
        prev_time = current_time

        # Run YOLOv8 inference on the frame
        # Use stream=True for efficient processing of video frames
        results = model.predict(source=frame, conf=conf_threshold, verbose=False, stream=True) 

        # Process results generator
        for res in results:
            # Draw bounding boxes on the frame
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]}: {conf:.2f}"
                
                # Get color based on class (example: green for empty, red for occupied)
                color = (0, 255, 0) if model.names[cls] == 'space-empty' else (0, 0, 255) 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps_calc:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the annotated frame
        cv2.imshow("Parking Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.") 