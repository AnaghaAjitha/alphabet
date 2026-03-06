import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


SAMPLE_RATE = 16000
DURATION = 1
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.0003

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
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

        if len(audio) < TARGET_LENGTH:
            audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
        else:
            audio = audio[:TARGET_LENGTH]

        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        # Gaussian Noise
        if self.training and np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise

        # Amplitude variation (helps L, W)
        if self.training and np.random.rand() < 0.3:
            audio = audio * np.random.uniform(0.8, 1.2)

        # Time Shift
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

        #High-frequency emphasis (helps S)
        if self.training and np.random.rand() < 0.3:
            mel_db[int(N_MELS*0.6):, :] *= 1.1

        # SpecAugment
        if self.training:
            #frequency masking
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


#attention model
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
#linear layer used to compute attention scores
        self.attention = nn.Linear(96 * 2, 1)

#randomly turns off 65% of neurons during training
        self.dropout = nn.Dropout(0.65)

#This is the final classifier.It converts the attention output vector into alphabet predictions.
        self.fc = nn.Linear(96 * 2, num_classes)

#how input data flows through the model.

# spectrogram initially looks like:
# (batch, mel, time)
# CNNs expect:
# (batch, channels, height, width)
# So we add a channel dimension:
# (batch, 1, mel, time)

    def forward(self, x):
        x = x.unsqueeze(1)
#extracts local frequency-time patterns from the spectrogram
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


#load data
train_dataset = SpeechDataset(TRAIN_PATH, training=True)
test_dataset = SpeechDataset(TEST_PATH, training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = AttentionModel(num_classes=len(train_dataset.label_map)).to(DEVICE)


#adding class weights and boosting
labels = train_dataset.labels
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Strong boost for plosives/fricatives
for letter in ['b', 'p', 'q', 'z','l', 'w', 's']:
    idx = train_dataset.label_map[letter]
    class_weights[idx] *= 1.6


criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# training loop
best_test_acc = 0
patience = 20
counter = 0

for epoch in range(EPOCHS):
    model.train()
    correct = total = 0

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
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
        torch.save(model.state_dict(), "attention_best_model_4.pth")
        print("New Best Model Saved!")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered!")
        break

print("Best Test Accuracy:", best_test_acc)

# CONFUSION MATRIX
model.load_state_dict(torch.load("attention_best_model_4.pth"))
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

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=alphabet_labels,
    yticklabels=alphabet_labels,
    cmap="Blues"
)
plt.title("Confusion Matrix (Focused Model)")
plt.show()
