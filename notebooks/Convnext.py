import os, json
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, callbacks, optimizers, losses
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Configuration ---
SPLIT_DIR   = r".\dataset_split"
OUTPUT_DIR  = r".\outputs_convnext"  # Updated folder name
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)
keras.utils.set_random_seed(SEED)

# --- Data Loading & Preprocessing ---
with open(os.path.join(SPLIT_DIR, "class_weights.json")) as f:
    class_weight_dict = {int(k): v for k, v in json.load(f).items()}

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=True, seed=SEED,
)
val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False,
)
test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False,
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
class_weights_tensor = tf.constant(
    [class_weight_dict[i] for i in range(NUM_CLASSES)], dtype=tf.float32
)

# --- Mixup & Utilities ---
def to_onehot(images, labels):
    return images, tf.one_hot(labels, NUM_CLASSES)

def mixup(images, labels, alpha=0.2):
    bs = tf.shape(images)[0]
    # Sample lambda from beta distribution
    lam = tf.random.gamma([bs], alpha)
    lam2 = tf.random.gamma([bs], alpha)
    lam = lam / (lam + lam2 + 1e-8)
    
    lam_img = tf.reshape(lam, [-1, 1, 1, 1])
    lam_lbl = tf.reshape(lam, [-1, 1])
    idx = tf.random.shuffle(tf.range(bs))
    
    return (lam_img * images + (1 - lam_img) * tf.gather(images, idx),
            lam_lbl * labels + (1 - lam_lbl) * tf.gather(labels, idx))

def add_sample_weight(images, labels):
    sw = tf.gather(class_weights_tensor, tf.argmax(labels, axis=-1))
    return images, labels, sw

AUTOTUNE = tf.data.AUTOTUNE
train_ds_mix = (
    train_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)
    .map(mixup, num_parallel_calls=AUTOTUNE)
    .map(add_sample_weight, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds_oh = val_ds.map(to_onehot, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# --- Model Building (ConvNeXt) ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom((-0.1, 0.1)),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Using ConvNeXtTiny for a good balance of speed and accuracy
# Note: ConvNeXt models in Keras typically expect input in range [0, 255] 
# as they have internal normalization layers.
backbone = keras.applications.ConvNeXtTiny(
    include_top=False, 
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3),
    pooling=None
)
backbone.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
# ConvNeXt handles its own preprocessing inside the architecture
x = backbone(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.LayerNormalization()(x) # Common practice for ConvNeXt
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="gelu")(x) # ConvNeXt uses GELU
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# --- Phase 1: Training the Head ---
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

history1 = model.fit(
    train_ds_mix, validation_data=val_ds_oh,
    epochs=PHASE1_EPOCHS,
    callbacks=[
        callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, "phase1_best.keras"), 
                                  monitor="val_accuracy", save_best_only=True),
        callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "phase1_log.csv"))
    ]
)

# --- Phase 2: Fine-Tuning ---
backbone.trainable = True
# Freeze the first 70% of layers
freeze_until = int(len(backbone.layers) * 0.7)
for layer in backbone.layers[:freeze_until]:
    layer.trainable = False

# ConvNeXt doesn't use standard BatchNorm; it uses LayerNorm, 
# which we usually keep trainable during fine-tuning.

steps_per_epoch = len(train_ds_mix)
lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5, # Lower LR for fine-tuning
    decay_steps=PHASE2_EPOCHS * steps_per_epoch,
    alpha=1e-6
)

model.compile(
    optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds_mix, validation_data=val_ds_oh,
    epochs=PHASE2_EPOCHS,
    callbacks=[
        callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, "phase2_best.keras"), 
                                  monitor="val_accuracy", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]
)

# --- Evaluation ---
test_ds_oh = test_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)
test_results = model.evaluate(test_ds_oh)
print(f"Test Accuracy: {test_results[1]:.4f}")