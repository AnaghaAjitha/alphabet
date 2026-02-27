import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


SAMPLE_RATE = 16000
DURATION = 1
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.0005

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpeechDataset(Dataset):
    def __init__(self, root_path, training=False):
        self.files = []
        self.labels = []
        self.training = training

        labels = sorted(os.listdir(root_path))
        self.label_map = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_map.items()}

        for label in labels:
            label_path = os.path.join(root_path, label)
            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(label_path, file))
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

        # Pad / Trim
        if len(audio) < TARGET_LENGTH:
            audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
        else:
            audio = audio[:TARGET_LENGTH]

        # Normalize amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        #Time Shift Augmentation
        if self.training:
            shift = np.random.randint(-1600, 1600)  # ±0.1 sec
            audio = np.roll(audio, shift)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        # SpecAugment
        if self.training:
            if np.random.rand() < 0.5:
                f = np.random.randint(5, 12)
                f0 = np.random.randint(0, mel_db.shape[0] - f)
                mel_db[f0:f0+f, :] = 0

            if np.random.rand() < 0.5:
                t = np.random.randint(5, 12)
                t0 = np.random.randint(0, mel_db.shape[1] - t)
                mel_db[:, t0:t0+t] = 0

        mel_db = torch.tensor(mel_db, dtype=torch.float32)
        return mel_db, label



class BalancedCNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=26):
        super(BalancedCNN_BiLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.MaxPool2d((2, 1))
        )

        self.lstm = nn.LSTM(
            input_size=64 * (N_MELS // 8),  # 64 * 8 = 512
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)

        batch, channels, mel, time = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * mel)

        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)

        return x

#LOAD DATA
train_dataset = SpeechDataset(TRAIN_PATH, training=True)
test_dataset = SpeechDataset(TEST_PATH, training=False)

print("Label Mapping:", train_dataset.label_map)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = BalancedCNN_BiLSTM(num_classes=len(train_dataset.label_map)).to(DEVICE)

# Class weights
labels = train_dataset.labels
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

#Label Smoothing Added
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

#Cosine LR Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


#EARLY STOPPING
best_test_acc = 0
patience = 8
counter = 0

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    scheduler.step()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        counter = 0
        torch.save(model.state_dict(), "final_best_model.pth")
        print("New Best Model Saved!")
    else:
        counter += 1
        print(f"No improvement for {counter} epochs")

    if counter >= patience:
        print("Early stopping triggered!")
        break

print("Best Test Accuracy:", best_test_acc)


#CONFUSION MATRIX

model.load_state_dict(torch.load("final_best_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
alphabet_labels = [train_dataset.idx_to_label[i] for i in range(len(train_dataset.idx_to_label))]

print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=alphabet_labels,
    yticklabels=alphabet_labels
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("Training complete.")
