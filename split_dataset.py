import os
import random
import shutil

# Paths
base_dir = "data"
train_input = os.path.join(base_dir, "train/input")
train_target = os.path.join(base_dir, "train/target")

val_input = os.path.join(base_dir, "val/input")
val_target = os.path.join(base_dir, "val/target")

test_input = os.path.join(base_dir, "test/input")
test_target = os.path.join(base_dir, "test/target")

# Create directories if not exist
for path in [val_input, val_target, test_input, test_target]:
    os.makedirs(path, exist_ok=True)

# Get all filenames
filenames = os.listdir(train_input)
filenames = [f for f in filenames if os.path.isfile(os.path.join(train_input, f))]

# Shuffle files
random.seed(42)  # reproducibility
random.shuffle(filenames)

# Split ratios
total = len(filenames)
val_size = int(0.1 * total)
test_size = int(0.1 * total)

val_files = filenames[:val_size]
test_files = filenames[val_size:val_size + test_size]

# Move validation files
for fname in val_files:
    shutil.move(os.path.join(train_input, fname), os.path.join(val_input, fname))
    shutil.move(os.path.join(train_target, fname), os.path.join(val_target, fname))

# Move test files
for fname in test_files:
    shutil.move(os.path.join(train_input, fname), os.path.join(test_input, fname))
    shutil.move(os.path.join(train_target, fname), os.path.join(test_target, fname))

print("Dataset split completed successfully!")
print(f"Training samples: {total - val_size - test_size}")
print(f"Validation samples: {val_size}")
print(f"Test samples: {test_size}")
