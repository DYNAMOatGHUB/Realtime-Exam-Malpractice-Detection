import os
import shutil
import tempfile
import argparse
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────
DEFAULT_MODEL = 'yolov8n-cls.pt'
DEFAULT_EPOCHS = 10
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 224

CLASS_MAP = {
    'normal act':     'NORMAL',
    'looking friend': 'LOOKING_AT_FRIEND',
    'giving object':  'GIVING_OBJECT',
    'giving code':    'GIVING_CODE',
    'cheating':       'CHEATING',
}

def clean_dataset(src_root: Path) -> Path:
    clean_dir = Path(tempfile.gettempdir()) / 'ExamAnomaly_Cleaned'
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True)
    
    print(f"Cleaning dataset from {src_root} -> {clean_dir}")
    
    for split in ['train', 'valid', 'val', 'test']:
        src_split = src_root / split
        if not src_split.exists():
            continue
        for orig, clean in CLASS_MAP.items():
            src_cls = src_split / orig
            if not src_cls.exists():
                print(f'  [Skipping] Not found: {split}/{orig}')
                continue
            
            # YOLOv8 expects "val" not "test" for classification training validation
            out_split = "val" if split == "test" else split
            dst_cls = clean_dir / out_split / clean
            
            shutil.copytree(src_cls, dst_cls, dirs_exist_ok=True)
            n = len(list(dst_cls.glob('*')))
            print(f'  [Copied] {split}/{orig} -> {out_split}/{clean} ({n} images)')
            
    # Auto-split validation if it doesn't exist
    train_dir = clean_dir / 'train'
    val_dir = clean_dir / 'val'
    if train_dir.exists() and not val_dir.exists():
        print("No validation set found. Auto-splitting 15% of train data...")
        import random
        for cls_dir in train_dir.iterdir():
            if not cls_dir.is_dir(): continue
            images = list(cls_dir.glob('*'))
            if not images: continue
            
            val_cls = val_dir / cls_dir.name
            val_cls.mkdir(parents=True, exist_ok=True)
            
            num_val = max(1, int(len(images) * 0.15))
            val_images = random.sample(images, num_val)
            
            for img in val_images:
                shutil.move(str(img), str(val_cls / img.name))
            
            print(f"  [Split] {cls_dir.name}: moved {num_val} images to val/")
            
    return clean_dir

def train(dataset_root: str, model_name: str, epochs: int, batch: int, imgsz: int):
    import torch
    from ultralytics import YOLO
    
    dataset_path = Path(dataset_root).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_path}")
        
    clean_root = clean_dataset(dataset_path)
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on device: {device}")
    if device == 'cpu':
        print("Warning: Training on CPU will be slow.")
        
    out_dir = Path(os.getcwd()) / 'examguard_output'
    
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)
    
    print("Starting YOLO training process...")
    results = model.train(
        data=str(clean_root),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_dir),
        name="yolo_cls_run",
        exist_ok=True
    )
    
    # Locate the best weights
    best_weights = out_dir / "yolo_cls_run" / "weights" / "best.pt"
    print(f"\nTraining complete! Best weights saved to: {best_weights}")
    
    # Try to copy weights automatically to ml/weights
    weights_dest = Path(__file__).resolve().parent.parent / "weights" / "exam_anomaly_classifier.pt"
    if best_weights.exists():
        weights_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, weights_dest)
        print(f"Copied weights to: {weights_dest}")
    else:
        print(f"Could not find best.pt at {best_weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Classification Model for Exam Anomalies")
    parser.add_argument("--dataset", type=str, required=True, help="Path to raw dataset folder")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model to use (e.g. yolov8n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image size")
    
    args = parser.parse_args()
    
    train(
        dataset_root=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
