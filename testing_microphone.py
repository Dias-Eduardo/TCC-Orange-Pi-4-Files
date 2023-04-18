import numpy as np

import pyaudio
import torchaudio
import torch
from SimpleCNN import SimpleCNN

import wave

num_classes = 9
input_shape = (12, 34)

# Para tirar as mensagens de erro quando usa PyAudio
from ctypes import *
from contextlib import contextmanager

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


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

with noalsaerr():
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

            # Grava o áudio em um .wav para debug
            wf = wave.open("audio.wav", "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(data)
            wf.close()

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
