# train_parking_spaces.py

from ultralytics import YOLO

def train_yolov8_model(data_yaml: str, pretrained_weights: str = "yolov8n.pt", epochs: int = 50, image_size: int = 640, device: int = 0):
    """
    Train a YOLOv8 model on your parking lot dataset using GPU if available.

    Parameters:
    - data_yaml: Path to the dataset YAML file.
    - pretrained_weights: Starting pre-trained model weights.
    - epochs: Number of training epochs.
    - image_size: Input image resolution.
    - device: GPU device id (0 by default). Use -1 for CPU.
    """
    # Create a YOLO model instance based on a pretrained model (nano version in this example)
    model = YOLO(pretrained_weights)
    
    # Start training with the specified GPU device
    results = model.train(data=data_yaml, epochs=epochs, imgsz=image_size, device=device)
    
    # After training, the best weights will be stored in runs/train/...
    print("Training complete. Check the 'runs/train' directory for results.")
    return model

if __name__ == '__main__':
    # Path to your data.yaml file
    data_yaml_path = "data.yaml"
    
    # Train the model using GPU (device=0 indicates the first GPU)
    trained_model = train_yolov8_model(data_yaml=data_yaml_path, device=0)
