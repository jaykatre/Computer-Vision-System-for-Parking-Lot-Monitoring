from flask import Flask, request, render_template, redirect, url_for, flash
# import requests # No longer needed
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
import traceback

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flash messages

# --- Configuration ---
# Path to your trained model weights. 
# Use the .pt file when running locally unless you have a specific reason/setup for .engine
model_path = "runs/detect/train9/weights/best.pt"
# model_path = "runs/detect/train/weights/best.engine" 

# Confidence threshold
conf_threshold = 0.5
# --- End Configuration ---

# --- Model Loading ---
# Load the YOLOv8 model (should load only once when the server starts)
try:
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    # Optional: Perform a dummy inference to initialize if needed
    # model.predict(np.zeros((640, 640, 3)), verbose=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Error loading model: {e}")
    print("Web app cannot function without the model.")
    # Depending on the desired behavior, you might want to exit or handle this differently.
    # For now, we'll set model to None and rely on checks in the route.
    model = None 
# --- End Model Loading ---

@app.route('/', methods=['GET'])
def index():
    # Render the main page with the upload form
    return render_template('index.html', processed_image=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        flash("Error: Model is not loaded. Cannot perform inference.")
        return redirect(url_for('index'))
        
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        try:
            # Read image file bytes
            img_bytes = file.read()
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                flash("Error: Could not decode image")
                return redirect(url_for('index'))

            print(f"Processing image: {frame.shape}")

            # Run YOLOv8 inference
            results = model.predict(source=frame, conf=conf_threshold, verbose=False, stream=False)
            
            # Assuming single image result
            res = results[0] 
            processed_frame = res.orig_img.copy() # Work on a copy

            # --- Calculate Counts ---
            empty_count = 0
            occupied_count = 0
            # --- End Calculate Counts ---

            # Draw bounding boxes on the frame
            for box in res.boxes:
                cls = int(box.cls[0])
                # Increment counts based on class index/name
                if model.names[cls] == 'space-empty':
                    empty_count += 1
                elif model.names[cls] == 'space-occupied':
                    occupied_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f"{model.names[cls]}: {conf:.2f}"
                color = (0, 255, 0) if model.names[cls] == 'space-empty' else (0, 0, 255) 
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode the processed frame to JPEG format in memory
            _, img_encoded = cv2.imencode('.jpg', processed_frame)
            
            # Convert bytes to base64 string
            base64_img_string = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

            print("Inference complete, rendering result.")
            print(f"Counts - Empty: {empty_count}, Occupied: {occupied_count}") # Log counts
            # Pass the base64 string and counts to the template
            return render_template('index.html', 
                                 processed_image=base64_img_string, 
                                 empty_count=empty_count, 
                                 occupied_count=occupied_count)

        except Exception as e:
            print(f"Error during processing: {e}")
            print("--- Full Traceback ---")
            traceback.print_exc() # Print the full traceback to console
            print("--- End Traceback ---")
            flash(f"An error occurred during processing: {e}")
            return redirect(url_for('index'))

    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist (Flask does this automatically, but good practice)
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Check if index.html exists - error if not, as it's crucial now
    if not os.path.exists('templates/index.html'):
         print("CRITICAL ERROR: templates/index.html not found! Please create it.")
         # You might want to exit here if the template is essential
         # For simplicity, we'll let Flask potentially error later
         
    if model is None:
         print("WARNING: Model failed to load. The application will run but inference will not work.")
            
    print("Starting Local Web Application Server...")
    # Use debug=True for development, False for production
    # Port 5000 is standard for Flask dev
    app.run(host='0.0.0.0', port=5000, debug=False) 