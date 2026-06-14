import os
import json
import numpy as np
import librosa
from pyts.image import RecurrencePlot
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import zoom

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
AUDIO_ORIGINAL = "dataset_final"
SAMPLE_RATE = 16000
DURATION = 1  
TARGET_SIZE = 64 

labels = sorted(os.listdir(AUDIO_ORIGINAL))
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

rp = RecurrencePlot(threshold='point', percentage=20)

def audio_to_cadence_matrix(file_path):
    """Carrega o áudio e gera a matriz de recorrência (cadência)"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    max_samples = SAMPLE_RATE * DURATION
    
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    else:
        audio = audio[:max_samples]
        
    audio_resized = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=2000)
    X_rp = rp.fit_transform(audio_resized.reshape(1, -1))
    
    scale_factor = TARGET_SIZE / X_rp.shape[1]
    matrix_cadence = zoom(X_rp[0], scale_factor, order=1)
    
    return np.expand_dims(matrix_cadence, axis=-1)

# ==========================================
# 2. CARREGAMENTO DOS DADOS (APENAS ORIGINAL PARA COMPARAÇÃO LIMPA)
# ==========================================
X, y = [], []
print("Extraindo gráficos de cadência para o Modelo CNN Puro...")
for label in labels:
    path = os.path.join(AUDIO_ORIGINAL, label)
    if not os.path.isdir(path): continue
    for file in os.listdir(path):
        if file.endswith(".wav"):
            try:
                matrix = audio_to_cadence_matrix(os.path.join(path, file))
                X.append(matrix)
                y.append(label_map[label])
            except Exception as e:
                pass

X = np.array(X)
y = np.array(y)

# Separação clássica de treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

# ==========================================
# 3. ARQUITETURA APENAS CNN (SEM LSTM)
# ==========================================
input_shape = X_train.shape[1:]

model_cnn_puro = Sequential([
    # Primeiro Bloco Convolucional
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Segundo Bloco Convolucional
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Terceiro Bloco Convolucional (para compensar a falta da LSTM na extração de features)
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Achatamento Espacial para o Classificador Denso
    Flatten(), 
    
    Dense(128, activation='relu'),
    Dropout(0.4),
    
    Dense(num_classes, activation='softmax')
])

model_cnn_puro.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(patience=4, factor=0.3)
]

# ==========================================
# 4. TREINAMENTO CENTRALIZADO PADRÃO
# ==========================================
print("\nIniciando Treinamento Centralizado da CNN Pura...")
history = model_cnn_puro.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=35,
    batch_size=32,
    callbacks=callbacks
)

# Salvando para a sua comparação
model_cnn_puro.save("modelo_cnn_puro_cadencia.h5")
print("\nModelo CNN Puro Salvo!")