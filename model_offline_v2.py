import os
import json
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# ============================
# CONFIGURATION
# ============================
IMG_DIR = "images"  # Folder containing images
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
CSV_FILE = "image_data.csv"  # CSV file with ID, component, defect
JSON_FILE = "image_data.json"  # JSON file with ID, component, defect
MODEL_PATH = "model.pth"  # Path to save/load model
PREDICTIONS_FILE = "predictions.csv"  # Output predictions file
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# ============================
# DEVICE SETUP
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# STEP 1: LOAD AND MERGE METADATA
# ============================
csv_data = pd.read_csv(CSV_FILE)

with open(JSON_FILE, "r") as f:
    json_data = json.load(f)

df_json = pd.DataFrame(json_data)
# metadata = df_json   #if json file is preferred
metadata = csv_data

# Split metadata
unique_ids = metadata["ID"].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)


train_df = metadata[metadata["ID"].isin(train_ids)]
test_df = metadata[metadata["ID"].isin(test_ids)]

# Move images
for img_id in train_df["ID"]:
    src = os.path.join(IMG_DIR, f"{img_id}.jpg")
    dst = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)

for img_id in test_df["ID"]:
    src = os.path.join(IMG_DIR, f"{img_id}.jpg")
    dst = os.path.join(TEST_DIR, f"{img_id}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)


# Function to filter rows based on existing images
def filter_existing_images(df, folder):
    existing_ids = {os.path.splitext(f)[0] for f in os.listdir(folder)}
    filtered_df = df[df["ID"].isin(existing_ids)]
    missing_ids = set(df["ID"]) - existing_ids
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs missing in {folder}. They will be removed.")
    return filtered_df

# Clean both CSVs
train_df_clean = filter_existing_images(train_df, TRAIN_DIR)
test_df_clean = filter_existing_images(test_df, TEST_DIR)

# Save cleaned CSVs
train_df_clean.to_csv("train_metadata.csv", index=False)
test_df_clean.to_csv("test_metadata.csv", index=False)

print(f"Cleaned train CSV: {len(train_df_clean)} rows")
print(f"Cleaned test CSV: {len(test_df_clean)} rows")

# shorter variable names
train_df = train_df_clean
test_df = test_df_clean

# ============================
# STEP 2: PREPARE MULTI-LABEL TARGETS
# ============================
train_grouped = train_df.groupby("ID").agg({
    "component": lambda x: list(set(x)),
    "defect": lambda x: list(set(x))
}).reset_index()

# Combine components and defects into one label set for easier training
train_grouped["labels"] = train_grouped.apply(lambda row: row["component"] + row["defect"], axis=1)

test_grouped = test_df.groupby("ID").agg({
    "component": lambda x: list(set(x)),
    "defect": lambda x: list(set(x))
}).reset_index()

test_grouped["labels"] = test_grouped.apply(lambda row: row["component"] + row["defect"], axis=1)

# Encode labels
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_grouped["labels"])
y_test = mlb.transform(test_grouped["labels"])
label_classes = mlb.classes_
print(f"Classes: {label_classes}")

# ============================
# STEP 3: CUSTOM DATASET
# ============================
class ImageDataset(Dataset):
    def __init__(self, img_dir, df, labels, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["ID"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, img_id
    
# ============================
# STEP 4: TRANSFORMS AND DATALOADERS
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create datasets
train_dataset = ImageDataset("images/train", train_grouped, y_train, transform=transform)
test_dataset = ImageDataset("images/test", test_grouped, y_test, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================
# STEP 5: DEFINE MODEL
# ============================
num_classes = len(label_classes)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Sigmoid()  # Multi-label output
)
model = model.to(device)

# ============================
# STEP 6: INCREMENTAL TRAINING SUPPORT
# ============================
if os.path.exists(MODEL_PATH):
    print("Loading existing model for incremental training...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# ============================
# STEP 7: TRAINING LOOP
# ============================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# Save model for future incremental training
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully.")

# ============================
# STEP 8: PREDICTION ON TEST SET AND SAVE TO CSV
# ============================
model.eval()
preds_all = []
predictions = []
with torch.no_grad():
    for images, _, img_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).int().cpu().numpy()
        preds_all.extend(preds)
        for i, img_id in enumerate(img_ids):
            pred_labels = [label_classes[j] for j in range(len(label_classes)) if preds[i][j] == 1]
            pred_components = [lbl for lbl in pred_labels if lbl in train_df["component"].unique()]
            pred_defects = [lbl for lbl in pred_labels if lbl in train_df["defect"].unique()]
            predictions.append({
                "ID": img_id,
                "components": ";".join(pred_components),
                "defects": ";".join(pred_defects)
            })

# Save predictions to CSV
pred_df = pd.DataFrame(predictions)

components_list = [
    "Centre Support","Channel Bracket","Circular Connection","Circular Joint",
    "Conveyor Support","Grout Hole","Radial Connection","Radial Joint","Walkway Support",
    "Centre Support-Bolt","Channel Bracket-Bolt","Conveyor Support-Bolt","Walkway Support-Bolt"
]

defects_list = [
    "Corrosion-Heavy","Coating Failure","Corrosion-Surface","Loose","Missing","Drummy","Leaks"
]

# Split the semicolon-separated string into a list
pred_df["split_values"] = pred_df["defects"].apply(lambda x: str(x).split(";") if isinstance(x, str) else [])

# Create new columns for components and defects
def split_into_categories(values):
    comp = [v for v in values if v in components_list]
    defect = [v for v in values if v in defects_list]
    return pd.Series([comp, defect])

pred_df[["components_clean", "defects_clean"]] = pred_df["split_values"].apply(split_into_categories)

# Convert lists to comma-separated strings for readability
pred_df["components_clean"] = pred_df["components_clean"].apply(lambda x: ", ".join(x))
pred_df["defects_clean"] = pred_df["defects_clean"].apply(lambda x: ", ".join(x))

# Drop helper column if not needed
pred_df.drop(columns=["split_values","components","defects"], inplace=True)

# Final DataFrame
print(pred_df.head())

# Save the cleaned predictions to a new CSV
pred_df.to_csv(PREDICTIONS_FILE, index=False)


# ============================
# STEP 9: Evaluation
# ============================
# F1 (micro) is most common for multi-label tasks because it accounts for imbalance.
# Hamming Loss is also useful for error rate.

# Convert predictions and true labels to arrays
true_labels = y_test  # from MultiLabelBinarizer
pred_labels_binary = preds_all  # collect all predictions from test_loader

# Hamming Loss
hl = hamming_loss(true_labels, pred_labels_binary)
print(f"Hamming Loss: {hl:.4f}")

# F1 Score (micro and macro)
f1_micro = f1_score(true_labels, pred_labels_binary, average='micro')
f1_macro = f1_score(true_labels, pred_labels_binary, average='macro')
print(f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")

# Subset Accuracy (strict match)
subset_acc = accuracy_score(true_labels, pred_labels_binary)
print(f"Subset Accuracy: {subset_acc:.4f}")
