import numpy as np

import pyaudio
import torchaudio
import torch
from SimpleCNN import SimpleCNN

num_classes = 9
input_shape = (12, 34)

n_mfcc = 12
n_fft = 512
hop_length = 480
n_mels = 40
center = True
norm = "ortho"


# O transform com a MFCC
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000, 
    n_mfcc=n_mfcc,
    norm=norm,
    melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "center": center},
)

# Carrega a CNN treinada anteriormente
cnn_model = SimpleCNN(num_classes=num_classes, input_shape=input_shape)
state_dict = torch.load('audio_model.pt')
cnn_model.load_state_dict(state_dict)  
cnn_model.eval()


labels =  ['backward', 'down', 'forward', 'left', 'off', 'on', 'right', 'stop', 'up']

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1000)
while True:
    # Pega uma parte do áudio do microfone e checa se ele possui volume
    data = stream.read(400)
    audio = np.frombuffer(data, dtype=np.int16)
    # print(np.abs(audio).mean())
    if np.abs(audio).mean() > 1000:
        # Pega um segundo inteiro de áudio
        data += stream.read(15600)
        # Transfere os dados adquiridos para tensor
        audio = np.frombuffer(data, dtype=np.int16)
        audio_float = audio.astype(np.float32) / np.iinfo(np.int16).max
        audio_tensor = torch.tensor(audio_float, dtype=torch.float32)
        # Extrai as features do audio adqurido
        mfcc_feats = mfcc_transform(audio_tensor)
        mfcc_feats = mfcc_feats.unsqueeze(0).unsqueeze(0)
        # Classifica o áudio com a CNN
        output = cnn_model(mfcc_feats)
        # Pega o resultado
        value, predicted = torch.max(output.data, 1)
        if value.item() > 4:
            print("Class:", labels[predicted.item()], "| Prediction Value =", str(value.item()))
        else:
            print("Unknown")
