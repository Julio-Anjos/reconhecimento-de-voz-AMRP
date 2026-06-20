import os
import json
import wave
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, LSTM, Dense, Dropout
)
from tensorflow.keras.losses import CategoricalCrossentropy

AUDIO_ORIGINAL = "dataset_final"
AUDIO_RUIDO = "dataset_ruido"
SAMPLE_RATE = 16000
DURATION = 1  
TARGET_SIZE = 64 

print("Verificando diretorios de dados...")
if not os.path.exists(AUDIO_ORIGINAL):
    raise FileNotFoundError(f"Pasta {AUDIO_ORIGINAL} nao encontrada.")

labels = sorted(os.listdir(AUDIO_ORIGINAL))
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)
print(f"Classes identificadas: {labels}")

def carregar_wav_manual(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        else:
            raise ValueError("Suporte apenas para WAV 16-bit PCM.")
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
        return audio_data, framerate

def extrair_espectrograma_linear_manual(audio, n_fft=512, hop_length=256):
    """
    Substitui o motor do Librosa. Executa uma Transformada de Fourier (STFT) 
    matematica pura em lotes usando apenas funcoes nativas do NumPy.
    """
    janela = np.hanning(n_fft)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames <= 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE))
        
    stft_matrix = []
    for t in range(num_frames):
        inicio = t * hop_length
        fim = inicio + n_fft
        fatia = audio[inicio:fim] * janela
        fft_valores = np.abs(np.fft.rfft(fatia))
        stft_matrix.append(fft_valores)
        
    stft_matrix = np.array(stft_matrix).T
    
    # Aplica compressao logaritmica de energia (escala Log-Espectrograma)
    stft_log = np.log1p(stft_matrix)
    return stft_log

def redimensionar_matriz_bilinear(matriz, target_size):
    orig_h, orig_w = matriz.shape
    if orig_h == 0 or orig_w == 0:
        return np.zeros((target_size, target_size))
        
    grid_h = np.linspace(0, orig_h - 1, target_size)
    grid_w = np.linspace(0, orig_w - 1, target_size)
    y_baicho = grid_h.astype(np.int32)
    y_alto = np.minimum(y_baicho + 1, orig_h - 1)
    x_esquerdo = grid_w.astype(np.int32)
    x_direito = np.minimum(x_esquerdo + 1, orig_w - 1)
    dy = (grid_h - y_baicho)[:, None]
    dx = (grid_w - x_esquerdo)[None, :]
    q11 = matriz[y_baicho[:, None], x_esquerdo]
    q12 = matriz[y_baicho[:, None], x_direito]
    q21 = matriz[y_alto[:, None], x_esquerdo]
    q22 = matriz[y_alto[:, None], x_direito]
    matriz_suave = (1 - dy) * (1 - dx) * q11 + (1 - dy) * dx * q12 + dy * (1 - dx) * q21 + dy * dx * q22
    return matriz_suave

def processar_pipeline_ideal(file_path):
    """
    Executa o pipeline acustico correto: Centraliza a energia por Volume Dinamico,
    extrai as componentes de frequencia lineares e normaliza os picos de cadencia.
    """
    audio, sr = carregar_wav_manual(file_path)
    
    # Remove as bordas mudas (VAD Manual por limiar de energia absoluta)
    limiar_energia = 0.05 * np.max(np.abs(audio))
    indices_uteis = np.where(np.abs(audio) > limiar_energia)[0]
    if len(indices_uteis) > 0:
        audio = audio[indices_uteis[0]:indices_uteis[-1]]
        
    max_samples = SAMPLE_RATE * DURATION
    if len(audio) < max_samples:
        total_pad = int(max_samples - len(audio))
        pad_esquerdo = total_pad // 2
        pad_direito = total_pad - pad_esquerdo
        audio = np.pad(audio, (pad_esquerdo, pad_direito), 'constant')
    else:
        audio = audio[:int(max_samples)]
        
    # Normalizacao por Maximo Absoluto (Protege contra variacoes de ganho)
    pico = np.max(np.abs(audio))
    if pico > 1e-6:
        audio = audio / pico
        
    espectrograma = extrair_espectrograma_linear_manual(audio)
    espectrograma_quadrado = redimensionar_matriz_bilinear(espectrograma, TARGET_SIZE)
    
    return np.expand_dims(espectrograma_quadrado, axis=-1)

X_orig, y_orig = [], []
print("\nIniciando extracao do dataset com o pipeline ideal...")
for label in labels:
    path = os.path.join(AUDIO_ORIGINAL, label)
    if not os.path.isdir(path): continue
    arquivos = [f for f in os.listdir(path) if f.endswith(".wav")]
    print(f"Processando classe [{label}] - Total de arquivos: {len(arquivos)}")
    for file in arquivos:
        try:
            matrix = processar_pipeline_ideal(os.path.join(path, file))
            X_orig.append(matrix)
            y_orig.append(label_map[label])
        except Exception as e:
            pass

X_orig = np.array(X_orig)
y_orig = np.array(y_orig)
print(f"Dataset original carregado. Formato: {X_orig.shape}")

X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42
)

X_train = X_train_orig
y_train = y_train_orig

if os.path.exists(AUDIO_RUIDO):
    X_aug, y_aug = [], []
    print("\nIniciando extracao do dataset com ruido...")
    for label in labels:
        path = os.path.join(AUDIO_RUIDO, label)
        if not os.path.isdir(path): continue
        arquivos_ruido = [f for f in os.listdir(path) if f.endswith(".wav")]
        print(f"Processando classe de ruido [{label}] - Total de arquivos: {len(arquivos_ruido)}")
        for file in arquivos_ruido:
            try:
                matrix = processar_pipeline_ideal(os.path.join(path, file))
                X_aug.append(matrix)
                y_aug.append(label_map[label])
            except:
                pass
    if len(X_aug) > 0:
        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)
        MAX_AUG_PER_CLASS = 800
        X_aug_bal, y_aug_bal = [], []
        for label in range(len(labels)):
            idx = np.where(y_aug == label)[0]
            if len(idx) > 0:
                np.random.shuffle(idx)
                idx = idx[:MAX_AUG_PER_CLASS]
                X_aug_bal.append(X_aug[idx])
                y_aug_bal.append(y_aug[idx])
        X_aug_bal = np.concatenate(X_aug_bal)
        y_aug_bal = np.concatenate(y_aug_bal)
        X_train = np.concatenate([X_train_orig, X_aug_bal])
        y_train = np.concatenate([y_train_orig, y_aug_bal])
        print(f"Dataset com ruido integrado e balanceado. Novo formato de treino: {X_train.shape}")

idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

input_shape = X_train.shape[1:]

print("\nConstruindo arquitetura CRNN Estavel...")
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Reshape(target_shape=(16, 16 * 64)), 
    LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

epochs_federadas = 35
batch_size = 32

chunks_X = np.array_split(X_train, 3)
chunks_y = np.array_split(y_train, 3)
total_amostras_treino = len(X_train)
proporcoes_nos = [len(chunks_X[i]) / total_amostras_treino for i in range(3)]

historico_loss = []
historico_acc = []

print("\nIniciando agregacao federada limpa (Weighted FedAvg)...")
for epoch in range(epochs_federadas):
    local_weights = []
    pesos_iniciais_epoca = model.get_weights()
    
    for node in range(3):
        model.set_weights(pesos_iniciais_epoca)
        model.fit(chunks_X[node], chunks_y[node], epochs=1, batch_size=batch_size, verbose=0)
        local_weights.append(model.get_weights())
        
    pesos_agregados = [np.zeros_like(w) for w in pesos_iniciais_epoca]
    for layer_idx in range(len(pesos_iniciais_epoca)):
        for node in range(3):
            pesos_agregados[layer_idx] += local_weights[node][layer_idx] * proporcoes_nos[node]
            
    model.set_weights(pesos_agregados)
    
    scores = model.evaluate(X_val, y_val, verbose=0)
    historico_loss.append(scores[0])
    historico_acc.append(scores[1])
    print(f"Epoca Federada Global [{epoch+1}/{epochs_federadas}] -> Loss Val: {scores[0]:.4f} | Acc Val: {scores[1]*100:.2f}%")

print("\nSalvando modelo finalizado...")
model.save("modelo_cadencia_lstm_federado_manual_2.h5")
with open("labels_cadencia_lstm.json", "w") as f:
    json.dump(label_map, f)

print("Exibindo graficos de evolucao...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_federadas + 1), historico_loss, color="red", marker="o")
plt.title("Evolucao do Loss Global (Validacao)")
plt.xlabel("Epoca Federada")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_federadas + 1), [a * 100 for a in historico_acc], color="blue", marker="s")
plt.title("Evolucao da Acuracia Global (Validacao)")
plt.xlabel("Epoca Federada")
plt.ylabel("Acuracia (%)")
plt.grid(True)

plt.tight_layout()
plt.show()