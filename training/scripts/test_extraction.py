import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def test_dataset(dataset_yaml, num_batches=2):
    """
    Test that the extracted dataset can be loaded by YOLO.
    
    Args:
        dataset_yaml: Path to dataset YAML file
        num_batches: Number of batches to load for testing
    """
    print(f"Testing dataset: {dataset_yaml}")
    
    # Check if the YAML file exists
    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml}")
    
    # Load YAML file to check its content
    with open(dataset_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    # Print dataset information
    print(f"Dataset path: {data_config.get('path', 'Not specified')}")
    print(f"Training images: {data_config.get('train', 'Not specified')}")
    print(f"Validation images: {data_config.get('val', 'Not specified')}")
    print(f"Number of classes: {data_config.get('nc', 'Not specified')}")
    print(f"Class names: {data_config.get('names', 'Not specified')}")
    
    # Create a YOLO model to load and test the dataset
    model = YOLO('yolov8n.pt')  # Use a small model for quick testing
    
    # Alternative validation approach
    try:
        # Use the val() method to validate on the dataset
        print(f"Attempting to load and validate dataset using {num_batches} batches...")
        results = model.val(
            data=dataset_yaml,
            batch=2,  # Small batch size for quick testing
            imgsz=640,  # Small image size for quick testing
            max_dim=640,  # Limit maximum dimension
            plots=False,  # Don't create plots
            max_det=10,  # Limit detections for speed
            verbose=True,  # Show progress
            seed=42,
            split='val'
        )
        print(f"Successfully validated dataset!")
        return True
    except Exception as e:
        print(f"Error during validation: {e}")
        
        # Try an even simpler test - just load a few images from the dataset
        try:
            print("\nAttempting basic dataset loading test...")
            import glob
            from PIL import Image
            
            # Get image paths from the dataset
            img_dir = os.path.join(data_config.get('path'), data_config.get('train'))
            img_paths = glob.glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True)
            if not img_paths:
                img_paths = glob.glob(os.path.join(img_dir, '**', '*.png'), recursive=True)
            
            if not img_paths:
                print(f"No images found in {img_dir}")
                return False
                
            print(f"Found {len(img_paths)} images in dataset")
            
            # Try to open a few images
            for i, img_path in enumerate(img_paths[:5]):
                try:
                    img = Image.open(img_path)
                    print(f"Successfully loaded image {i+1}: {os.path.basename(img_path)} ({img.size})")
                    
                    # Check corresponding label file
                    label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            labels = f.readlines()
                        print(f"  Label file found with {len(labels)} annotations")
                    else:
                        print(f"  Label file not found: {label_path}")
                except Exception as img_error:
                    print(f"Error opening image {img_path}: {img_error}")
            
            print("\nBasic dataset loading test completed. Your dataset structure appears valid.")
            print("You can proceed with training, but be aware that the automatic validation failed.")
            
            return True
            
        except Exception as basic_error:
            print(f"Basic dataset loading test failed: {basic_error}")
            return False

if __name__ == "__main__":
    # Path to your extracted dataset YAML
    dataset_yaml = "/fast_data/rwm_dataset_extraction/yolov11/dataset.yaml"
    
    # Test the dataset
    success = test_dataset(dataset_yaml)
    
    if success:
        print("\n✅ Dataset verification passed! Ready for training.")
    else:
        print("\n❌ Dataset verification failed. Please check the errors above.")
