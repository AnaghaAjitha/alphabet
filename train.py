import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#parameters
SAMPLE_RATE = 16000
DURATION = 1
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.0003

DATASET_PATH = "dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
class SpeechDataset(Dataset):

    def __init__(self, files, labels, label_map, training=False):
        self.files = files
        self.labels = labels
        self.training = training
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = self.files[idx]
        label = self.labels[idx]

        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

        if len(audio) < TARGET_LENGTH:
            audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
        else:
            audio = audio[:TARGET_LENGTH]

        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        # Gaussian Noise
        if self.training and np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise

        # Amplitude variation
        if self.training and np.random.rand() < 0.3:
            audio = audio * np.random.uniform(0.8, 1.2)

        # Time shift
        if self.training:
            shift = np.random.randint(-1600, 1600)
            audio = np.roll(audio, shift)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        # High-frequency emphasis
        if self.training and np.random.rand() < 0.3:
            mel_db[int(N_MELS*0.6):, :] *= 1.1

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


#attention_model
class AttentionModel(nn.Module):

    def __init__(self, num_classes=26):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.MaxPool2d((2, 1))
        )

        self.lstm = nn.LSTM(
            input_size=64 * (N_MELS // 8),
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(96 * 2, 1)

        self.dropout = nn.Dropout(0.65)

        self.fc = nn.Linear(96 * 2, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.cnn(x)

        batch, channels, mel, time = x.size()

        x = x.permute(0, 3, 1, 2)

        x = x.reshape(batch, time, channels * mel)

        x, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(x), dim=1)

        x = torch.sum(attn_weights * x, dim=1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


#loading wav data
files = []
labels = []

alphabet = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(alphabet)}
idx_to_label = {idx: label for label, idx in label_map.items()}

for label in alphabet:

    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            files.append(os.path.join(folder, file))
            labels.append(label_map[label])


#80-20 dataset split
train_files, test_files, train_labels, test_labels = train_test_split(
    files,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

#printing split count
train_count = {label:0 for label in alphabet}
test_count = {label:0 for label in alphabet}

for l in train_labels:
    train_count[idx_to_label[l]] += 1

for l in test_labels:
    test_count[idx_to_label[l]] += 1

print("\n--- TRAIN SPLIT ---")
for k in alphabet:
    print(f"{k} : {train_count[k]}")

print("\n--- TEST SPLIT ---")
for k in alphabet:
    print(f"{k} : {test_count[k]}")


#dataset
train_dataset = SpeechDataset(train_files, train_labels, label_map, training=True)
test_dataset = SpeechDataset(test_files, test_labels, label_map, training=False)
#dataset loading
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
#model loading
model = AttentionModel(num_classes=len(label_map)).to(DEVICE)


#assigning weights to each character
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
#boosting weaker words
for letter in ['b','d','p','q','z','v','l','w','s']:
    idx = label_map[letter]
    class_weights[idx] *= 1.6

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

#training loop
best_test_acc = 0
patience = 20
counter = 0

for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

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

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.3f} | Test: {test_acc:.3f}")

    if test_acc > best_test_acc:

        best_test_acc = test_acc
        counter = 0

        torch.save(model.state_dict(), "attention_best_model_8.pth")

        print("New Best Model Saved!")

    else:
        counter += 1

    if counter >= patience:

        print("Early stopping triggered!")
        break


print("Best Test Accuracy:", best_test_acc)


#confusion matrix
model.load_state_dict(torch.load("attention_best_model_8.pth"))

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

plt.figure(figsize=(14,12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=alphabet_labels,
    yticklabels=alphabet_labels,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.show()
