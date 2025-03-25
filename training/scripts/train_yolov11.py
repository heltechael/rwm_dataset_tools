from ultralytics import YOLO
import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on RWM dataset')
    parser.add_argument('--data', type=str, default='/fast_data/rwm_dataset_yolov11/dataset.yaml', 
                        help='Path to dataset configuration')
    parser.add_argument('--model', type=str, default='/fast_data/rwm_dataset_yolov11/models/yolo11x.pt', 
                        help='Path to model')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=6, 
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=1280, 
                        help='Image size')
    parser.add_argument('--device', type=str, default='0', 
                        help='Device(s) to use for training (comma-separated)')
    parser.add_argument('--workers', type=int, default=8, 
                        help='Number of worker threads')
    parser.add_argument('--name', type=str, default='yolov11_rwm', 
                        help='Name for the training run')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Convert device string to list if multiple devices
    devices = [int(x) for x in args.device.split(',')] if ',' in args.device else args.device
    
    # Load the model
    model = YOLO(args.model)
    
    # Try to use FocalLoss if available
    try:
        from ultralytics.utils.loss import FocalLoss
        model.loss = FocalLoss()
        print("Using FocalLoss")
    except (ImportError, AttributeError):
        print("FocalLoss not available, using default loss")
    
    # Train the model
    results = model.train(
        verbose=True,
        data=args.data,
        epochs=args.epochs,
        workers=args.workers,
        augment=True,
        val=True,
        save_period=50,
        close_mosaic=0,
        patience=0,
        cache=False,
        imgsz=args.img_size,
        rect=True,
        max_det=1500,
        cls=1.0,
        batch=args.batch_size,
        optimizer="auto",
        device=devices,
        project="runs/detection/",
        name=args.name,
        # Augmentations
        hsv_h=0.1,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.15,
        flipud=0.5,
        fliplr=0.5,
    )
    
    print(f"Training completed. Results saved to {results}")

if __name__ == "__main__":
    main()
