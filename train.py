import os
from pathlib import Path
from fastai.vision.all import *
def main():
    print("Initializing\n")
    # 1. SETUP TARGET DATASET PATH
    # Points to the exact extracted folder location on your Windows Desktop
    dataset_path=Path(r"C:\Users\prakr\OneDrive\Desktop\CODES\python programming\Projects\pneumonia_detector\chest_xray")
    if not dataset_path.exists():
        print("\nError: Could not find the dataset\n")
        return
    # 2. CONFIGURE THE PIPELINE (DATABLOCK)
    print("\nConfiguring\n")
    pneumonia_block=DataBlock(
        blocks=(ImageBlock, CategoryBlock),                   # Input: Images; Output: Text Labels
        get_items=get_image_files,                           # Recursively fetch all image formats (.jpg, .jpeg)
        splitter=GrandparentSplitter(train_name='train', valid_name='val'), # Splitting based on Kaggle's explicit Train/Val folders
        get_y=parent_label,                                  # Class label extracted from parent folder name (NORMAL/PNEUMONIA)
        item_tfms=Resize(224),                               # Standardize all chest X-rays to 224x224 pixels for ResNet50
        batch_tfms=aug_transforms(mult=1.0, do_flip=False)   # Data augmentations (flips disabled for medical symmetry)
    )
    # 3. BUILD DATALOADERS (CREATING BATCHES)
    print("Loading images into memory batches...")
    dls=pneumonia_block.dataloaders(dataset_path, bs=32, num_workers=0)
    print("Data successfully loaded!")
    # 4. INITIALIZE DEEP LEARNING MODEL (TRANSFER LEARNING)
    print("\nDownloading and assembling\n")
    # Tracks accuracy and error rate metrics during training loops
    learn=vision_learner(dls, resnet50, metrics=[accuracy, error_rate])
    # 5. RUN TRAINING LOOPS (FINE-TUNING)
    print("\nTraining started. Please wait while the model fits layers...\n")
    # 3 epochs is ideal for a fast resume project—gives high accuracy without taking hours
    learn.fine_tune(epochs=3, base_lr=2e-3)
    # 6. EXPORT WEIGHTS FOR PRODUCTION & GITHUB
    print("\nFinalizing training process...\n")
    model_filename='pneumonia_resnet50_model.pkl'
    learn.export(model_filename)
    print("SUCCESS! Your trained model is saved.")
if __name__ == "__main__":
    main()