# train_model.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import pathlib

# --------------------------
# Load and preprocess EuroSAT dataset
# --------------------------
def load_eurosat_dataset(data_dir: str):
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
        
    image_size = (64, 64)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class directories found in {data_dir}")
        
    X, y = [], []
    for idx, cls in enumerate(classes):
        cls_dir = data_dir / cls
        images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))
        if not images:
            print(f"Warning: No images found in class directory {cls}")
            continue
            
        for img_path in images:
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
                arr = tf.keras.preprocessing.image.img_to_array(img)
                X.append(arr)
                y.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                
    if not X:
        raise ValueError("No valid images found in the dataset")
        
    X = np.array(X) / 255.0
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    return X, y, classes

def preprocess_data(X, y):
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    # Use relative path and environment variable for data directory
    default_data_dir = os.path.join(os.path.dirname(__file__), 'EuroSAT_RGB', 'EuroSAT_RGB')
    DATA_DIR = os.getenv('EUROSAT_DATA_DIR', default_data_dir)
    
    # Create save directory if it doesn't exist
    save_dir = pathlib.Path("saved_model")
    save_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {DATA_DIR}")
    X, y, classes = load_eurosat_dataset(DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)
    
    # Build and train model with callbacks
    model = build_cnn_model(X_train.shape[1:], len(classes))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "best_model.h5"),
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=8,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(save_dir / "satellite_cnn_model.h5")
    
    # Evaluation
    print("\nEvaluating model on test set:")
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=classes))
    
    # Save classes list
    with open(save_dir / "classes.txt", "w") as f:
        f.write("\n".join(classes))

if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as e:
        print(f"Error during model training: {e}")



# # train_model_densenet.py
# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.applications.densenet import preprocess_input
# from sklearn.metrics import classification_report
# import pathlib

# # --------------------------
# # Load and preprocess EuroSAT dataset
# # --------------------------
# def load_eurosat_dataset(data_dir: str):
#     data_dir = pathlib.Path(data_dir)
#     if not data_dir.exists():
#         raise ValueError(f"Data directory not found: {data_dir}")

#     image_size = (64, 64)
#     classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
#     if not classes:
#         raise ValueError(f"No class directories found in {data_dir}")

#     X, y = [], []
#     for idx, cls in enumerate(classes):
#         cls_dir = data_dir / cls
#         images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))
#         if not images:
#             print(f"Warning: No images found in class directory {cls}")
#             continue

#         for img_path in images:
#             try:
#                 img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
#                 arr = tf.keras.preprocessing.image.img_to_array(img)
#                 X.append(arr)
#                 y.append(idx)
#             except Exception as e:
#                 print(f"Error loading image {img_path}: {e}")

#     if not X:
#         raise ValueError("No valid images found in the dataset")

#     X = np.array(X)
#     X = preprocess_input(X)  # For DenseNet
#     y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
#     return X, y, classes

# def preprocess_data(X, y):
#     X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test

# def build_densenet_model(input_shape, num_classes):
#     base_model = DenseNet121(
#         include_top=False,
#         weights='imagenet',
#         input_shape=input_shape,
#         pooling='avg'
#     )
#     base_model.trainable = False  # Freeze base model

#     model = models.Sequential([
#         base_model,
#         layers.BatchNormalization(),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation='softmax')
#     ])

#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def train_and_save_model():
#     default_data_dir = os.path.join(os.path.dirname(__file__), 'EuroSAT_RGB', 'EuroSAT_RGB')
#     DATA_DIR = os.getenv('EUROSAT_DATA_DIR', default_data_dir)

#     save_dir = pathlib.Path("saved_model_densenet")
#     save_dir.mkdir(exist_ok=True)

#     print(f"Loading data from {DATA_DIR}")
#     X, y, classes = load_eurosat_dataset(DATA_DIR)
#     X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

#     model = build_densenet_model(X_train.shape[1:], len(classes))

#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#         tf.keras.callbacks.ModelCheckpoint(filepath=str(save_dir / "best_model.h5"),
#                                            monitor='val_accuracy', save_best_only=True),
#         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
#     ]

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=15,
#         batch_size=32,
#         callbacks=callbacks
#     )

#     model.save(save_dir / "satellite_densenet_model.h5")

#     print("\nEvaluating model on test set:")
#     y_pred = model.predict(X_test)
#     y_true = np.argmax(y_test, axis=1)
#     y_pred_labels = np.argmax(y_pred, axis=1)

#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred_labels, target_names=classes))

#     with open(save_dir / "classes.txt", "w") as f:
#         f.write("\n".join(classes))

# if __name__ == "__main__":
#     try:
#         train_and_save_model()
#     except Exception as e:
#         print(f"Error during model training: {e}")
