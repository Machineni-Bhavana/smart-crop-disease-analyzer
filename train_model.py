import os
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import ssl

# Fix macOS local issuer certificate errors for urllib/Keras weights download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# pylint: disable=no-member, no-name-in-module, import-error
from tensorflow.keras import layers, models, applications, callbacks  # type: ignore

# Global Class Names matching the disease info
CLASS_NAMES = [
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Scab",
    "Corn Common Rust",
    "Healthy",
    "Tomato Blight",
]

# Kaggle to Local Name mappings for automated extraction
KAGGLE_MAPPING = {
    "Apple___Black_rot": "Apple Black Rot",
    "Apple___Cedar_apple_rust": "Apple Cedar Rust",
    "Apple___Apple_scab": "Apple Scab",
    "Corn_(maize)___Common_rust_": "Corn Common Rust",
    "Tomato___Early_blight": "Tomato Blight",
    "Tomato___Late_blight": "Tomato Blight",
    "Apple___healthy": "Healthy",
    "Corn_(maize)___healthy": "Healthy",
    "Tomato___healthy": "Healthy"
}


def load_real_dataset(data_dir, batch_size=32, image_size=(224, 224)):
    """
    Loads dataset from directory and splits into Train (70%), Val (15%), Test (15%).
    If empty, invokes kagglehub to orchestrate and parse dataset remotely to data/.
    """
    import shutil
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if we have our 6 folders populated
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    if len(folders) < 6:
        print("Data directory is missing our target classes. Using kagglehub to retrieve PlantVillage dataset...")
        try:
            import kagglehub # type: ignore
            print("Downloading 'abdallahalidev/plantvillage-dataset' via kagglehub (~2GB)... This will take a few minutes!")
            path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
            
            # The structure is usually path/color or path/plantvillage dataset/color
            color_dir = os.path.join(path, "plantvillage dataset", "color")
            if not os.path.exists(color_dir):
                color_dir = os.path.join(path, "color") # alternative architecture
                
            print(f"Dataset downloaded to {path}. Extracting matching classes into data/...")
            
            for folder_name in os.listdir(color_dir):
                if folder_name in KAGGLE_MAPPING:
                    target_class = KAGGLE_MAPPING[folder_name]
                    src_folder = os.path.join(color_dir, folder_name)
                    dest_folder = os.path.join(data_dir, target_class)
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    # Move images
                    for file in os.listdir(src_folder):
                        src_file = os.path.join(src_folder, file)
                        dest_file = os.path.join(dest_folder, f"{folder_name}_{file}")
                        if not os.path.exists(dest_file):
                            shutil.copy2(src_file, dest_file)
                            
            print("✅ Successfully filtered and extracted target classes into data/!")
        except Exception as e:
            print(f"Failed to download/parse dataset automatically: {e}")
            return None, None, None

    print(f"Loading real dataset from {data_dir}...")

    try:
        # 85% for train/val, 15% for test
        train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.15,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
        )

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.15,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
        )
    except ValueError as e:
        print(f"Dataset loading failed (likely empty or missing valid files): {e}")
        return None, None, None

    # Split train_val_ds into 70% train, 15% val
    # train_val_ds has 85% of total.
    # to get 15% of total as val, we take 15/85 ~ 17.6% of train_val_ds
    val_batches = int(len(train_val_ds) * 0.176)
    val_ds = train_val_ds.take(val_batches)
    train_ds = train_val_ds.skip(val_batches)

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_transfer_learning_model(
    input_shape=(224, 224, 3), num_classes=6, learning_rate=1e-4
):
    """
    Builds a MobileNetV2 based model for transfer learning.
    """
    # Base model MobileNetV2
    base_model = applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    # Freeze the base model
    base_model.trainable = False

    model = models.Sequential(
        [
            # Data Augmentation layer
            tf.keras.Sequential(
                [
                    layers.RandomFlip("horizontal_and_vertical"),
                    layers.RandomRotation(0.2),
                    layers.RandomZoom(0.2),
                ],
                name="data_augmentation",
            ),
            # Preprocessing expected by MobileNetV2 ranges [-1, 1] usually, but Keras application can handle it or we rescale:
            layers.Rescaling(1.0 / 127.5, offset=-1),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_dummy_data(num_samples=120):
    X = np.random.rand(num_samples, 224, 224, 3).astype("float32")
    y = np.array([i % 6 for i in range(num_samples)])
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(16)
    return ds


def train_model():
    data_dir = "data"
    model_dir = "models"
    model_path = os.path.join(model_dir, "crop_disease_model.keras")
    os.makedirs(model_dir, exist_ok=True)

    print("1. Preparing data...")
    train_ds, val_ds, test_ds = load_real_dataset(data_dir)
    num_classes = len(CLASS_NAMES)

    print("2. Building MobileNetV2 Transfer Learning Model...")
    model = build_transfer_learning_model(num_classes=num_classes)
    model.summary()

    # Callbacks: Early Stopping and Checkpointing
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        model_path, monitor="val_accuracy", save_best_only=True
    )

    print("3. Starting Training Process...")
    if train_ds is not None:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=[early_stop, checkpoint],
        )
        # Evaluate on test set
        print("Evaluating on Test Set...")
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test Accuracy: {test_acc:.2%}")

    else:
        print(
            "Training on dummy dataset purely to instantiate and create model.keras...."
        )
        dummy_ds = create_dummy_data()
        model.fit(dummy_ds, epochs=1, callbacks=[checkpoint])

    print("Training complete.")


if __name__ == "__main__":
    train_model()
