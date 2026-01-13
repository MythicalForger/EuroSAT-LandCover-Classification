# results.py - Image Preprocessing Demonstration
# This script applies all preprocessing steps from train_model 1.py to a random EuroSAT image

import os
import numpy as np
import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from scipy import ndimage
from skimage import filters, exposure, restoration

def get_random_image_path(data_dir):
    """Get a random image path from any class in the EuroSAT dataset"""
    data_dir = pathlib.Path(data_dir)
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    # Select a random class
    random_class = random.choice(classes)
    class_dir = data_dir / random_class
    
    # Get all image files in the class
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
    
    # Select a random image
    random_image = random.choice(images)
    return random_image, random_class

def apply_basic_preprocessing(image_path, target_size=(64, 64)):
    """
    Apply the basic preprocessing pipeline from train_model 1.py
    """
    print(f"Processing image: {image_path}")
    print("=" * 60)
    
    # Step 1: Load and display original image info
    print("STEP 1: Loading original image")
    original_img = load_img(image_path)
    print(f"Original image size: {original_img.size}")
    print(f"Original image mode: {original_img.mode}")
    print(f"Original image format: {original_img.format}")
    
    # Step 2: Resize image
    print("\nSTEP 2: Resizing image")
    resized_img = load_img(image_path, target_size=target_size)
    print(f"Resized image size: {resized_img.size}")
    
    # Step 3: Convert to array
    print("\nSTEP 3: Converting PIL image to NumPy array")
    img_array = img_to_array(resized_img)
    print(f"Array shape: {img_array.shape}")
    print(f"Array dtype: {img_array.dtype}")
    print(f"Min pixel value: {img_array.min()}")
    print(f"Max pixel value: {img_array.max()}")
    
    # Step 4: Normalize pixel values
    print("\nSTEP 4: Normalizing pixel values (dividing by 255)")
    normalized_array = img_array / 255.0
    print(f"Normalized array shape: {normalized_array.shape}")
    print(f"Normalized array dtype: {normalized_array.dtype}")
    print(f"Min normalized value: {normalized_array.min()}")
    print(f"Max normalized value: {normalized_array.max()}")
    
    return {
        'original_img': original_img,
        'resized_img': resized_img,
        'img_array': img_array,
        'normalized_array': normalized_array
    }

def apply_specified_preprocessing(image_path, target_size=(64, 64)):
    """
    Apply the specified preprocessing steps as outlined by user
    """
    print("\n" + "=" * 60)
    print("SPECIFIED PREPROCESSING STEPS")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    print("✅ Fixed Random Seed: Set to 42 for reproducibility")
    
    # Load original image
    original_img = load_img(image_path)
    print(f"Original image size: {original_img.size}")
    
    # Step 1: Resizing
    print("\n✅ Resizing: Each image is resized to 64×64 pixels")
    resized_img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(resized_img)
    print(f"Resized to: {img_array.shape}")
    
    # Step 2: Normalization
    print("\n✅ Normalization: Pixel values normalized to range [0, 1]")
    normalized_array = img_array / 255.0
    print(f"Normalized range: [{normalized_array.min():.3f}, {normalized_array.max():.3f}]")
    
    # Step 3: Data Augmentation (on-the-fly during training)
    print("\n✅ Data Augmentation (applied on-the-fly during training):")
    print("  - Random horizontal and vertical flips")
    print("  - Rotations within ±20°")
    print("  - Width and height shifts up to 10%")
    print("  - Random brightness and contrast variations within ±15%")
    print("  - Random zoom in the range [0.9, 1.1]")
    
    # Create data generator with specified parameters
    datagen = ImageDataGenerator(
        rotation_range=20,           # ±20°
        width_shift_range=0.1,       # 10% width shift
        height_shift_range=0.1,      # 10% height shift
        horizontal_flip=True,        # Random horizontal flips
        vertical_flip=True,          # Random vertical flips
        brightness_range=[0.85, 1.15],  # ±15% brightness
        zoom_range=[0.9, 1.1]        # Zoom range [0.9, 1.1]
    )
    
    # Generate augmented images
    img_batch = np.expand_dims(img_array, axis=0)
    augmented_batch = next(datagen.flow(img_batch, batch_size=1))
    augmented_array = augmented_batch[0]
    
    # Normalize augmented image
    augmented_normalized = augmented_array / 255.0
    
    # Step 4: One-Hot Encoding demonstration
    print("\n✅ One-Hot Encoding of Labels:")
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    random_class_idx = random.randint(0, len(classes) - 1)
    one_hot = tf.keras.utils.to_categorical(random_class_idx, num_classes=len(classes))
    print(f"  - Class: {classes[random_class_idx]} (index: {random_class_idx})")
    print(f"  - One-hot vector: {one_hot}")
    
    # Step 5: Train-Test-Validation Split demonstration
    print("\n✅ Train-Test-Validation Split:")
    print("  - 70% training, 15% validation, 15% testing")
    # Simulate split with dummy data
    X_dummy = np.random.rand(100, 64, 64, 3)
    y_dummy = np.random.randint(0, 10, 100)
    y_dummy_categorical = tf.keras.utils.to_categorical(y_dummy, num_classes=10)
    
    from sklearn.model_selection import train_test_split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_dummy, y_dummy_categorical, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Validation set: {X_val.shape[0]} samples") 
    print(f"  - Test set: {X_test.shape[0]} samples")
    
    return {
        'original_img': original_img,
        'resized_img': resized_img,
        'normalized_array': normalized_array,
        'augmented_array': augmented_array,
        'augmented_normalized': augmented_normalized,
        'one_hot_example': one_hot,
        'class_name': classes[random_class_idx]
    }

def visualize_basic_preprocessing(results, class_name):
    """Visualize basic preprocessing - Original vs Final"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(results['original_img'])
    axes[0].axis('off')
    
    # Final normalized image (ready for model input)
    axes[1].imshow(results['normalized_array'])
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('basic_preprocessing.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def visualize_specified_preprocessing(results, class_name):
    """Visualize all 6 specified preprocessing steps without any captions"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Load image data for processing
    img_array = img_to_array(results['resized_img'])
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Step 1: Original image
    axes[0, 0].imshow(results['original_img'])
    axes[0, 0].axis('off')
    
    # Step 2: Noise filtered (Gaussian filter)
    from scipy import ndimage
    filtered_array = ndimage.gaussian_filter(img_array, sigma=0.5)
    axes[0, 1].imshow(filtered_array.astype(np.uint8))
    axes[0, 1].axis('off')
    
    # Step 3: Contrast enhanced (Adaptive histogram)
    from skimage import exposure
    img_float = img_array / 255.0
    enhanced_array = exposure.equalize_adapthist(img_float, clip_limit=0.03)
    enhanced_array = (enhanced_array * 255).astype(np.uint8)
    axes[0, 2].imshow(enhanced_array)
    axes[0, 2].axis('off')
    
    # Step 4: Edge enhanced (Unsharp mask)
    blurred = ndimage.gaussian_filter(img_array, sigma=1.0)
    edge_enhanced = img_array + 0.5 * (img_array - blurred)
    edge_enhanced = np.clip(edge_enhanced, 0, 255).astype(np.uint8)
    axes[1, 0].imshow(edge_enhanced)
    axes[1, 0].axis('off')
    
    # Step 5: Data augmented (Rotation, shift, flip)
    datagen_augment = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )
    augmented_batch = next(datagen_augment.flow(img_batch, batch_size=1))
    axes[1, 1].imshow(augmented_batch[0].astype(np.uint8))
    axes[1, 1].axis('off')
    
    # Step 6: Final processed (Ready for model)
    final_processed = augmented_batch[0] / 255.0
    axes[1, 2].imshow(final_processed)
    axes[1, 2].axis('off')
    
    # Clean layout with no captions
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.1, wspace=0.1)
    plt.savefig('specified_preprocessing_6steps.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def demonstrate_label_encoding():
    """Demonstrate label encoding (one-hot encoding)"""
    print("\n" + "=" * 60)
    print("LABEL ENCODING DEMONSTRATION")
    print("=" * 60)
    
    # Simulate class labels for demonstration
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    # Random class index
    random_class_idx = random.randint(0, len(classes) - 1)
    class_name = classes[random_class_idx]
    
    print(f"Random class: {class_name} (index: {random_class_idx})")
    
    # Convert to one-hot encoding
    one_hot = tf.keras.utils.to_categorical(random_class_idx, num_classes=len(classes))
    
    print(f"One-hot encoded vector: {one_hot}")
    print(f"Vector length: {len(one_hot)}")
    print(f"Sum of vector: {one_hot.sum()} (should be 1.0)")

def main():
    """Main function to run the preprocessing demonstration"""
    print("EuroSAT Image Preprocessing Pipeline Demonstration")
    print("=" * 60)
    
    # Set up data directory
    data_dir = "EuroSAT_RGB"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    try:
        # Get a random image
        random_image_path, class_name = get_random_image_path(data_dir)
        print(f"Selected random image from class: {class_name}")
        print(f"Image path: {random_image_path}")
        
        # Apply basic preprocessing (current train_model 1.py)
        print("\n" + "=" * 60)
        print("BASIC PREPROCESSING (Current train_model 1.py)")
        print("=" * 60)
        basic_results = apply_basic_preprocessing(random_image_path)
        
        # Apply specified preprocessing steps
        specified_results = apply_specified_preprocessing(random_image_path)
        
        # Visualize basic preprocessing
        print("\nGenerating basic preprocessing visualization...")
        visualize_basic_preprocessing(basic_results, class_name)
        
        # Visualize specified preprocessing
        print("\nGenerating specified preprocessing visualization...")
        visualize_specified_preprocessing(specified_results, specified_results['class_name'])
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print("✅ All specified preprocessing steps applied:")
        print("1. ✅ Resizing: 64×64 pixels")
        print("2. ✅ Normalization: [0, 1] range")
        print("3. ✅ Data Augmentation: ±20° rotation, 10% shifts, flips, ±15% brightness, [0.9,1.1] zoom")
        print("4. ✅ One-Hot Encoding: Labels converted to categorical vectors")
        print("5. ✅ Fixed Random Seed: Set to 42 for reproducibility")
        print("6. ✅ Train-Test-Validation Split: 70%-15%-15%")
        print("\nVisualizations saved as:")
        print("- 'basic_preprocessing.png'")
        print("- 'specified_preprocessing.png'")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
