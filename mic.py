
import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd


# ===============================
# PARAMETERS (same as training)
# ===============================
SAMPLE_RATE = 16000
DURATION = 1
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
idx_to_label = {i: alphabet[i] for i in range(26)}


# ===============================
# ATTENTION MODEL
# ===============================
class AttentionModel(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.MaxPool2d((2,1))
        )

        self.lstm = nn.LSTM(
            input_size=64 * (N_MELS // 8),
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(96 * 2, 1)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(96 * 2, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.cnn(x)

        batch, channels, mel, time = x.size()

        x = x.permute(0,3,1,2)
        x = x.reshape(batch, time, channels * mel)

        x, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


# ===============================
# LOAD MODEL
# ===============================
model = AttentionModel(num_classes=26).to(DEVICE)
model.load_state_dict(torch.load("attention_best_model_4.pth", map_location=DEVICE))
model.eval()

print("Model loaded successfully!")
print("Press ENTER to record")
print("Type 'q' then ENTER to quit.\n")


# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess_audio(audio):

    if len(audio) < TARGET_LENGTH:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
    else:
        audio = audio[:TARGET_LENGTH]

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    mel_db = torch.tensor(mel_db, dtype=torch.float32)
    mel_db = mel_db.unsqueeze(0).to(DEVICE)

    return mel_db


# ===============================
# MIC LOOP
# ===============================
while True:

    user_input = input("Press ENTER to record: ")

    if user_input.lower() == "q":
        print("Exiting.")
        break

    print("🎙 Recording... Speak now")

    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()
    input_tensor = preprocess_audio(audio)

    with torch.no_grad():

        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

    predicted_letter = idx_to_label[pred.item()]
    conf_percent = confidence.item() * 100

    print(f"Predicted: {predicted_letter.upper()} | Confidence: {conf_percent:.2f}%\n")
