import os
import sys
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from collections import deque
from mtcnn import MTCNN 
import numpy as np
from tqdm import tqdm
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --------------------------- USER-CONFIG (NO NEED to CHANGE normally) ---------------------------
DATASET_DIR = r"C:\Users\sowbi\OneDrive\Desktop\project-Robotics\archive"  # <--- your provided path
MODEL_FILE = os.path.join(os.getcwd(), "stress_model.h5")
CLASS_MAP_FILE = os.path.join(os.getcwd(), "class_map.json")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
CAMERA_INDEX = 0
SMOOTHING_WINDOW = 10   # sliding window for prediction stability
BINARY_TEMP_DIR = os.path.join(os.getcwd(), "binary_dataset")
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
# -----------------------------------------------------------------------------------------------

# Keywords used to automatically map folder names to 'stressed' or 'not_stressed'
STRESS_KEYWORDS = {"fear", "afraid", "scared", "angry", "anger", "disgust", "sad", "sadness", "alone", "lonely", "panic", "anxious", "anxiety", "stressed", "stress", "frustrated", "frustration"}
NOT_STRESS_KEYWORDS = {"happy", "joy", "smile", "neutral", "calm", "content", "surprise", "surprised", "relaxed", "pleased", "satisfied"}

# -----------------------------------------------------------------------------------------------

def list_subfolders(path):
    path = Path(path)
    if not path.exists():
        return []
    all_folders = []
    for sub in path.rglob('*'):
        if sub.is_dir() and any(sub.glob('*')):  # has files
            # Skip root folder itself
            if sub != path:
                all_folders.append(str(sub.relative_to(path)))
    return all_folders


def keyword_map_classname(name: str):
    """Return 'stressed' or 'not_stressed' or None based on keywords in name."""
    n = name.lower()
    for k in STRESS_KEYWORDS:
        if k in n:
            return 'stressed'
    for k in NOT_STRESS_KEYWORDS:
        if k in n:
            return 'not_stressed'
    return None


def build_binary_dataset(original_root, stressed_folders, notstressed_folders, out_root=BINARY_TEMP_DIR):
    """
    Create binary dataset: stressed/ and not_stressed/ by copying images
    from all stressed_folders and notstressed_folders, even if they are inside train/test.
    """
    # Check if the output folder exists, reuse or create it
    if os.path.exists(out_root):
        print(f"Reusing existing temporary binary dataset at {out_root}")
    else:
        os.makedirs(out_root, exist_ok=True)
        print(f"Created new temporary binary dataset at {out_root}")

    # Set up stressed and not_stressed subfolders
    stress_dir = os.path.join(out_root, 'stressed')
    not_dir = os.path.join(out_root, 'not_stressed')
    os.makedirs(stress_dir, exist_ok=True)
    os.makedirs(not_dir, exist_ok=True)

    # Function to copy images with progress and limit
    def copy_from_folders(folder_list, target_dir):
        copied = 0
        max_images = 1000  # Limit to 1000 images per class for speed
        total_files = sum(len([f for f in os.listdir(os.path.join(original_root, fp)) if f.lower().endswith(SUPPORTED_EXTS)]) for fp in folder_list if os.path.isdir(os.path.join(original_root, fp)))
        print(f"Copying up to {min(total_files, max_images)} images to {target_dir}...")
        with tqdm(total=min(total_files, max_images), desc=f"Copying to {target_dir}") as pbar:
            for folder_path in folder_list:
                src_dir = os.path.join(original_root, folder_path)
                if not os.path.isdir(src_dir):
                    print(f"Skipping invalid folder: {src_dir}")
                    continue
                for fname in os.listdir(src_dir)[:max_images]:  # Limit images
                    if fname.lower().endswith(SUPPORTED_EXTS):
                        src = os.path.join(src_dir, fname)
                        dst = os.path.join(target_dir, f"{folder_path.replace(os.sep,'__')}__{fname}")
                        try:
                            shutil.copy2(src, dst)
                            copied += 1
                            pbar.update(1)
                            if copied >= max_images:
                                break
                        except Exception as e:
                            print(f"Failed to copy {src}: {e}")
                if copied >= max_images:
                    break
        return copied

    # Copy images to respective folders
    c1 = copy_from_folders(stressed_folders, stress_dir)
    c2 = copy_from_folders(notstressed_folders, not_dir)
    print(f"Copied {c1} images to {stress_dir}")
    print(f"Copied {c2} images to {not_dir}")

    # Verify the dataset is valid before returning
    if not os.path.exists(stress_dir) or not os.path.exists(not_dir) or c1 == 0 or c2 == 0:
        raise ValueError(f"Binary dataset creation failed! Check folders {stress_dir} and {not_dir}")
    
    return out_root  # Always return the path

def build_binary_model(input_shape=(224,224,3)):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = True  # Allow learning
    for layer in base.layers[:-20]:  # Freeze all but last 20 layers
        layer.trainable = False  # Balance speed and accuracy
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=preds)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_multiclass_model(num_classes, input_shape=(224,224,3)):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = True  # Allow learning
    for layer in base.layers[:-20]:  # Freeze all but last 20 layers
        layer.trainable = False  # Balance speed and accuracy
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=preds)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_binary(binary_dir, model_out=MODEL_FILE, epochs=EPOCHS, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    print("Training binary classifier from:", binary_dir)
    datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.18,
    rotation_range=30,  # More tilt for head angles
    width_shift_range=0.15,  # Slide left-right
    height_shift_range=0.15,  # Slide up-down
    shear_range=0.15,  # Stretch for weird angles
    zoom_range=0.15,  # Zoom in/out for distance
    brightness_range=[0.8, 1.2],  # Handle glasses glare or dim light
    horizontal_flip=True,
    fill_mode='nearest')  # Fill gaps when twisting

    train_gen = datagen.flow_from_directory(binary_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='training')
    val_gen = datagen.flow_from_directory(binary_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='validation')

    model = build_binary_model(input_shape=(image_size[0], image_size[1], 3))
    callbacks = [
        ModelCheckpoint(model_out, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    model.save(model_out)
    print(f"Saved binary model to {model_out}")
    return model


def train_multiclass(dataset_root, model_out=MODEL_FILE, epochs=EPOCHS, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    print("Training multiclass classifier from:", dataset_root)
    datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.18,
    rotation_range=30,  # More tilt for head angles
    width_shift_range=0.15,  # Slide left-right
    height_shift_range=0.15,  # Slide up-down
    shear_range=0.15,  # Stretch for weird angles
    zoom_range=0.15,  # Zoom in/out for distance
    brightness_range=[0.8, 1.2],  # Handle glasses glare or dim light
    horizontal_flip=True,
    fill_mode='nearest')  # Fill gaps when twisting

    train_gen = datagen.flow_from_directory(dataset_root, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(dataset_root, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')
    num_classes = train_gen.num_classes
    model = build_multiclass_model(num_classes=num_classes, input_shape=(image_size[0], image_size[1], 3))
    callbacks = [
        ModelCheckpoint(model_out, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    model.save(model_out)
    # Save the class indices -> this is important to map prediction idx -> class name
    class_indices = {v:k for k,v in train_gen.class_indices.items()}  # invert mapping to idx->name
    return model, class_indices


# -------------------- LIVE DETECTION / INFERENCE --------------------

def run_live_inference(model, mode, class_map, camera_index=CAMERA_INDEX, image_size=IMAGE_SIZE, smoothing=SMOOTHING_WINDOW):
    """
    mode: 'binary' or 'multiclass'
    class_map: when binary -> {'stressed_folders': [...], 'notstressed_folders':[...]} OR when multiclass -> mapping idx->class_name
    """
    detector = MTCNN()  # Initialize MTCNN detector
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}")
        return # Return if camera fails to open

    prob_buffer = deque(maxlen=smoothing)
    print("Starting live camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        # Start of processing logic, now correctly inside the loop
        display_frame = frame.copy()
        results = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if len(results) == 0:
            cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            for result in results:
                # Ensure detection confidence is high enough
                if result['confidence'] < 0.95:
                    continue

                x, y, w, h = result['box']
                
                # Expand bounding box for better context
                pad = int(0.12 * w)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face = frame[y1:y2, x1:x2]

                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, image_size)
                    inp = face_resized.astype('float32') / 255.0
                    inp = np.expand_dims(inp, axis=0)
                except Exception:
                    # Skip this face if resizing or color conversion fails
                    continue

                # ---- MODEL PREDICTION LOGIC ----
                if mode == 'binary':
                    p = float(model.predict(inp)[0][0])
                    prob_buffer.append(p)
                    avg_p = float(np.mean(prob_buffer)) if prob_buffer else p
                    label = 'STRESSED' if avg_p >= 0.7 else 'NOT STRESSED'
                    text = f"{label} ({avg_p:.2f})"
                    color = (0, 0, 255) if label == 'STRESSED' else (0, 255, 0)
                else:  # multiclass
                    preds = model.predict(inp)[0]
                    idx = int(np.argmax(preds))
                    class_name = class_map.get(str(idx), class_map.get(idx, "unknown"))
                    
                    mapping = keyword_map_classname(str(class_name))
                    is_stressed = (mapping == 'stressed')
                    prob_buffer.append(1.0 if is_stressed else 0.0)
                    avg_p = float(np.mean(prob_buffer)) if prob_buffer else 0.0
                    
                    label = 'STRESSED' if avg_p >= 0.7 else 'NOT STRESSED'
                    text = f"{label} ({avg_p:.2f}) [{class_name}]"
                    color = (0, 0, 255) if label == 'STRESSED' else (0, 255, 0)

                # Draw bounding box and label text on the display frame
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the final frame
        cv2.imshow('Live Stress Detector', display_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


# ------------------------- MAIN FLOW -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATASET_DIR, help='root folder of dataset (class subfolders)')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX, help='camera index (0 default)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='training epochs')
    parser.add_argument('--mode', choices=['auto','train'], default='auto', help='auto = run (train if needed), train = force training')
    parser.add_argument('--force-train', action='store_true', help='force retrain even if model exists')
    args = parser.parse_args()

    dataset_root = os.path.normpath(args.dataset)
    if not os.path.isdir(dataset_root):
        print(f"Dataset path not found: {dataset_root}")
        sys.exit(1)

    # discover folders
    folders = list_subfolders(dataset_root)
    if len(folders) == 0:
        print("No class subfolders found in dataset root. Make sure your images are inside class subfolders.")
        sys.exit(1)

    print("Found class folders:", folders)

    stressed_folders = []
    notstressed_folders = []
    unmapped = []
    for f in folders:
        m = keyword_map_classname(Path(f).name)  # check only the last folder name
        if m == 'stressed':
            stressed_folders.append(f)
        elif m == 'not_stressed':
            notstressed_folders.append(f)
        else:
            unmapped.append(f)


    print(f"Auto-mapped stressed folders: {stressed_folders}")
    print(f"Auto-mapped not-stressed folders: {notstressed_folders}")
    if len(unmapped) > 0:
        print(f"Unmapped (ignored for binary mapping): {unmapped}")

    # Decide training strategy
    binary_possible = (len(stressed_folders) > 0 and len(notstressed_folders) > 0)

    model = None
    class_map = {}

    if os.path.exists(MODEL_FILE) and not args.force_train and args.mode == 'auto':
        try:
            print(f"Loading existing model from {MODEL_FILE}")
            model = load_model(MODEL_FILE)
            # attempt to load class_map.json
            if os.path.exists(CLASS_MAP_FILE):
                with open(CLASS_MAP_FILE, 'r') as f:
                    class_map = json.load(f)
            # determine mode: if class_map has binary flag or keys
            if 'binary' in class_map.get('mode','') or binary_possible:
                mode_type = 'binary' if binary_possible or 'binary' in class_map.get('mode','') else 'multiclass'
            else:
                mode_type = 'multiclass'
            print(f"Model loaded. Running in {mode_type} mode.")
            run_live_inference(model, mode_type, class_map, camera_index=args.camera)
            return
        except Exception as e:
            print("Failed to load existing model (will retrain). Reason:", e)

    # If training forced or no model present, train
    if args.mode == 'train' or args.force_train or (not os.path.exists(MODEL_FILE)):
        if binary_possible:
            # create binary dataset and train binary classifier
            binary_dir = build_binary_dataset(dataset_root, stressed_folders, notstressed_folders)
            model = train_binary(binary_dir, model_out=MODEL_FILE, epochs=args.epochs)
            class_map = {
                'mode': 'binary',
                'stressed_folders': stressed_folders,
                'notstressed_folders': notstressed_folders
            }
            with open(CLASS_MAP_FILE, 'w') as f:
                json.dump(class_map, f)
            print("Binary training complete. Starting live detection...")
            run_live_inference(model, 'binary', class_map, camera_index=args.camera)
            return
        else:
            # fallback: train multiclass on all folders and use keyword mapping at inference
            model, idx_to_class = train_multiclass(dataset_root, model_out=MODEL_FILE, epochs=args.epochs)
            # save idx->class map
            with open(CLASS_MAP_FILE, 'w') as f:
                json.dump({str(k): v for k,v in idx_to_class.items()}, f)
            class_map = {str(k): v for k,v in idx_to_class.items()}
            print("Multiclass training complete. Starting live detection (will map classes to stressed/neutral by keyword).")
            run_live_inference(model, 'multiclass', class_map, camera_index=args.camera)
            return

    # fallback - should not reach here normally
    print("Nothing to do. Exiting.")


if __name__ == '__main__':
    main()
