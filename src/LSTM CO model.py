# import os
# import json
# import numpy as np
# import librosa
# from pyts.image import RecurrencePlot
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv2D, MaxPooling2D, BatchNormalization,
#     Reshape, LSTM, Dense, Dropout
# )
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# from sklearn.preprocessing import label_binarize
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # ==========================================
# # 1. CONFIGURAÇÕES E CAMINHOS
# # ==========================================
# # Mudamos para ler as pastas de áudio direto, visando calcular a cadência
# AUDIO_ORIGINAL = "dataset_final"
# AUDIO_RUIDO = "dataset_ruido"

# SAMPLE_RATE = 16000
# DURATION = 1  # Fixando 1 segundo para manter as matrizes simétricas
# TARGET_SIZE = 64 # Dimensão da Matriz de Cadência (64x64)

# labels = sorted(os.listdir(AUDIO_ORIGINAL))
# label_map = {label: i for i, label in enumerate(labels)}
# num_classes = len(labels)

# # Inicializador do Gráfico de Recorrência (Cadência)
# rp = RecurrencePlot(threshold='point', percentage=20)

# def audio_to_cadence_matrix(file_path):
#     """Carrega o áudio e gera a matriz de recorrência (cadência)"""
#     audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#     max_samples = SAMPLE_RATE * DURATION
    
#     # Padroniza tamanho
#     if len(audio) < max_samples:
#         audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
#     else:
#         audio = audio[:max_samples]
        
#     # Downsampling rápido para não estourar a memória com matrizes gigantescas
#     audio_resized = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=2000)
    
#     # Transforma o vetor 1D na matriz de cadência 2D
#     X_rp = rp.fit_transform(audio_resized.reshape(1, -1))
    
#     # Redimensiona para um tamanho quadrado padrão (ex: 64x64) para a CNN
#     from scipy.ndimage import zoom
#     scale_factor = TARGET_SIZE / X_rp.shape[1]
#     matrix_cadence = zoom(X_rp[0], scale_factor, order=1)
    
#     return np.expand_dims(matrix_cadence, axis=-1)

# # ==========================================
# # 2. CARREGAMENTO E PROCESSAMENTO DOS DADOS
# # ==========================================
# X_orig, y_orig = [], []

# print("Extraindo gráficos de cadência do dataset original...")
# for label in labels:
#     path = os.path.join(AUDIO_ORIGINAL, label)
#     if not os.path.isdir(path): continue

#     for file in os.listdir(path):
#         if file.endswith(".wav"):
#             try:
#                 matrix = audio_to_cadence_matrix(os.path.join(path, file))
#                 X_orig.append(matrix)
#                 y_orig.append(label_map[label])
#             except Exception as e:
#                 pass

# X_orig = np.array(X_orig)
# y_orig = np.array(y_orig)

# X_train_orig, X_val, y_train_orig, y_val = train_test_split(
#     X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42
# )

# X_aug, y_aug = [], []
# print("Extraindo gráficos de cadência do dataset com ruído...")
# for label in labels:
#     path = os.path.join(AUDIO_RUIDO, label)
#     if not os.path.isdir(path): continue

#     for file in os.listdir(path):
#         if file.endswith(".wav"):
#             try:
#                 matrix = audio_to_cadence_matrix(os.path.join(path, file))
#                 X_aug.append(matrix)
#                 y_aug.append(label_map[label])
#             except Exception as e:
#                 pass

# X_aug = np.array(X_aug)
# y_aug = np.array(y_aug)

# # Balanceamento artificial do ruído
# MAX_AUG_PER_CLASS = 800
# X_aug_bal, y_aug_bal = [], []

# for label in range(len(labels)):
#     idx = np.where(y_aug == label)[0]
#     if len(idx) > 0:
#         np.random.shuffle(idx)
#         idx = idx[:MAX_AUG_PER_CLASS]
#         X_aug_bal.append(X_aug[idx])
#         y_aug_bal.append(y_aug[idx])

# X_aug_bal = np.concatenate(X_aug_bal)
# y_aug_bal = np.concatenate(y_aug_bal)

# X_train = np.concatenate([X_train_orig, X_aug_bal])
# y_train = np.concatenate([y_train_orig, y_aug_bal])

# # Permutação
# idx = np.random.permutation(len(X_train))
# X_train = X_train[idx]
# y_train = y_train[idx]

# y_train = to_categorical(y_train, num_classes=num_classes)
# y_val = to_categorical(y_val, num_classes=num_classes)

# print("Formato do Treino (Cadência):", X_train.shape)
# print("Formato da Validação (Cadência):", X_val.shape)

# # ==========================================
# # 3. ARQUITETURA CRNN (CNN + LSTM)
# # ==========================================
# input_shape = X_train.shape[1:]

# model = Sequential([
#     # Extração de Feat. Geométricas da Cadência via CNN
#     Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), padding='same', activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     # Preparação para a Recorrência da LSTM
#     # Achata as dimensões espaciais mantendo o eixo temporal intacto
#     Reshape(target_shape=(16, 16 * 64)), 

#     # Camada Recorrente de Memória (LSTM) com forte proteção a overfitting
#     LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
#     BatchNormalization(),

#     Dense(64, activation='relu'),
#     Dropout(0.4),

#     Dense(num_classes, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss=CategoricalCrossentropy(label_smoothing=0.05), # Evita overfit/underfit dando margem a incertezas
#     metrics=['accuracy']
# )

# callbacks = [
#     EarlyStopping(patience=6, restore_best_weights=True),
#     ReduceLROnPlateau(patience=3, factor=0.3)
# ]

# # ==========================================
# # 4. SIMULAÇÃO DE TREINAMENTO COLABORATIVO (FEDAVG SIMULADO)
# # ==========================================
# # Para mimetizar o Collaborative Learning sem infraestrutura de rede, 
# # fatiamos o dataset em 3 "nós locais fictícios", treinamos de forma independente
# # por lote e agregamos a média dos pesos de volta ao modelo global para generalização.
# print("\n[Collaborative Learning] Iniciando agregação simulada de pesos por Nós...")
# epochs_federadas = 35
# batch_size = 32

# history_loss, history_val_loss = [], []
# history_acc, history_val_acc = [], []

# for epoch in range(epochs_federadas):
#     # Divide os dados de treino para simular 3 dispositivos independentes
#     chunks_X = np.array_split(X_train, 3)
#     chunks_y = np.array_split(y_train, 3)
    
#     local_weights = []
    
#     # Cada "Nó" treina de forma isolada sobre a cadência da sua própria partição
#     for node in range(3):
#         model.fit(chunks_X[node], chunks_y[node], epochs=1, batch_size=batch_size, verbose=0)
#         local_weights.append(model.get_weights())
        
#     # Agregação Global (FedAvg) -> Tira a média aritmética de todos os pesos aprendidos
#     avg_weights = [np.mean([weights[i] for weights in local_weights], axis=0) for i in range(len(local_weights[0]))]
#     model.set_weights(avg_weights)
    
#     # Avaliação da época federada global
#     scores = model.evaluate(X_val, y_val, verbose=0)
#     train_scores = model.evaluate(X_train, y_train, verbose=0)
    
#     history_loss.append(train_scores[0])
#     history_val_loss.append(scores[0])
#     history_acc.append(train_scores[1])
#     history_val_acc.append(scores[1])
    
#     print(f"Época Federada Global [{epoch+1}/{epochs_federadas}] -> Loss Val: {scores[0]:.4f} | Acc Val: {scores[1]*100:.2f}%")

# # ==========================================
# # 5. SALVAMENTO DO MODELO
# # ==========================================
# model.save("modelo_cadencia_lstm_federado.h5")

# with open("labels_cadencia_lstm.json", "w") as f:
#     json.dump(label_map, f)
# print("\nModelo Híbrido Federado Salvo!")

# # ==========================================
# # 6. GERAÇÃO DE GRÁFICOS DE DESEMPENHO
# # ==========================================
# preds = model.predict(X_val)
# y_pred = np.argmax(preds, axis=1)
# y_true = np.argmax(y_val, axis=1)

# # 1. Matriz de Confusão
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
# plt.xlabel("Predito")
# plt.ylabel("Real")
# plt.title("Matriz de Confusão (Cadência + LSTM)")
# plt.show()

# # 2. Histórico Dinâmico de Curvas de Aprendizado
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history_loss, label='Treino (Média Global)')
# plt.plot(history_val_loss, label='Validação')
# plt.title("Evolução do Loss Global")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history_acc, label='Treino (Média Global)')
# plt.plot(history_val_acc, label='Validação')
# plt.title("Evolução da Acurácia Global")
# plt.legend()
# plt.show()

# # 3. Curva ROC Multi-classe
# y_val_bin = label_binarize(y_true, classes=range(num_classes))
# plt.figure(figsize=(8, 6))
# for i in range(num_classes):
#     fpr, tpr, _ = roc_curve(y_val_bin[:, i], preds[:, i])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f'Classe {labels[i]} (AUC={roc_auc:.2f})')
# plt.plot([0,1],[0,1],'k--')
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.title("Curva ROC (Abordagem Dinâmica Não-Linear)")
# plt.legend()
# plt.show()

# # 4. Relatório Clínico de Métricas (Precision/Recall/F1)
# report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
# df = pd.DataFrame(report).transpose()
# df[['precision','recall','f1-score']].iloc[:-3].plot(kind='bar', figsize=(10,5), color=['#1f77b4', '#aec7e8', '#ff7f0e'])
# plt.title("Métricas de Validação por Classe")
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--')
# plt.show()

# # 5. Análise de Separação de Espaço de Características (PCA das predições da LSTM)
# pca = PCA(n_components=2)
# reduced = pca.fit_transform(preds)
# plt.figure(figsize=(8,6))
# scatter = plt.scatter(reduced[:,0], reduced[:,1], c=y_true, cmap='viridis', alpha=0.7)
# plt.title("Separação Geométrica de Comandos (PCA nas saídas da LSTM)")
# plt.colorbar(scatter, ticks=range(num_classes), format=plt.FuncFormatter(lambda val, loc: labels[int(val)]))
# plt.show()

import os
import json
import wave
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, LSTM, Dense, Dropout
)
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


AUDIO_ORIGINAL = "dataset_final"
AUDIO_RUIDO = "dataset_ruido"

SAMPLE_RATE = 16000
DURATION = 1  
TARGET_SIZE = 64 

labels = sorted(os.listdir(AUDIO_ORIGINAL))
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)



def carregar_wav_manual(file_path):
    """Abre arquivos .wav mono de 16-bit PCM usando a biblioteca nativa wave"""
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Leitura dos bytes brutos
        raw_data = wav_file.readframes(n_frames)
        
        # Converte de bytes estruturados de 16-bit para inteiros
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        else:
            raise ValueError("O script suporta apenas arquivos WAV de 16-bit PCM.")
            
       
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
            
        return audio_data, framerate

def resize_matriz_manual(matriz, target_size):
    """Redimensiona uma matriz 2D de forma manual usando amostragem por vizinho mais próximo"""
    orig_h, orig_w = matriz.shape
    # Cria uma grade com as novas coordenadas mapeadas para o tamanho antigo
    row_indices = (np.arange(target_size) * (orig_h / target_size)).astype(np.int32)
    col_indices = (np.arange(target_size) * (orig_w / target_size)).astype(np.int32)
    

    return matriz[row_indices[:, None], col_indices]

def calcular_cadencia_manual(audio, threshold_percentage=0.2):
    """Calcula a matriz de recorrência (cadência) de forma puramente matemática"""
  
    audio_2k = audio[::8]
    matriz_distancias = np.abs(audio_2k[:, None] - audio_2k[None, :])
    
    limiar = np.percentile(matriz_distancias, threshold_percentage * 100)
    matriz_recorrencia = (matriz_distancias <= limiar).astype(np.float32)
    
    matriz_redimensionada = resize_matriz_manual(matriz_recorrencia, TARGET_SIZE)
    
    return np.expand_dims(matriz_redimensionada, axis=-1)

def audio_to_cadence_matrix(file_path):
    """Pipeline unificado sem uso de Librosa ou Pyts"""
    audio, sr = carregar_wav_manual(file_path)
    max_samples = SAMPLE_RATE * DURATION
    
    # Padroniza tamanho fixo do áudio em 1 segundo
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, int(max_samples - len(audio))), 'constant')
    else:
        audio = audio[:int(max_samples)]
        
    # Normalização de amplitude do sinal (Z-score manual) para mitigar variações de ganho de microfone
    if np.std(audio) > 0:
        audio = (audio - np.mean(audio)) / np.std(audio)
        
    return calcular_cadencia_manual(audio, threshold_percentage=0.2)


X_orig, y_orig = [], []

print("Extraindo gráficos de cadência manuais do dataset original...")
for label in labels:
    path = os.path.join(AUDIO_ORIGINAL, label)
    if not os.path.isdir(path): continue

    for file in os.listdir(path):
        if file.endswith(".wav"):
            try:
                matrix = audio_to_cadence_matrix(os.path.join(path, file))
                X_orig.append(matrix)
                y_orig.append(label_map[label])
            except Exception as e:
                pass

X_orig = np.array(X_orig)
y_orig = np.array(y_orig)

X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42
)

X_aug, y_aug = [], []
print("Extraindo gráficos de cadência manuais do dataset com ruído...")
for label in labels:
    path = os.path.join(AUDIO_RUIDO, label)
    if not os.path.isdir(path): continue

    for file in os.listdir(path):
        if file.endswith(".wav"):
            try:
                matrix = audio_to_cadence_matrix(os.path.join(path, file))
                X_aug.append(matrix)
                y_aug.append(label_map[label])
            except Exception as e:
                pass

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

idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)


input_shape = X_train.shape[1:]

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

print("\n[Collaborative Learning] Iniciando agregação de pesos...")
epochs_federadas = 35
batch_size = 32

history_loss, history_val_loss = [], []
history_acc, history_val_acc = [], []

for epoch in range(epochs_federadas):
    chunks_X = np.array_split(X_train, 3)
    chunks_y = np.array_split(y_train, 3)
    
    local_weights = []
    for node in range(3):
        model.fit(chunks_X[node], chunks_y[node], epochs=1, batch_size=batch_size, verbose=0)
        local_weights.append(model.get_weights())
        
    avg_weights = [np.mean([weights[i] for weights in local_weights], axis=0) for i in range(len(local_weights[0]))]
    model.set_weights(avg_weights)
    
    scores = model.evaluate(X_val, y_val, verbose=0)
    train_scores = model.evaluate(X_train, y_train, verbose=0)
    
    history_loss.append(train_scores[0])
    history_val_loss.append(scores[0])
    history_acc.append(train_scores[1])
    history_val_acc.append(scores[1])
    
    print(f"Época Federada Global [{epoch+1}/{epochs_federadas}] -> Loss Val: {scores[0]:.4f} | Acc Val: {scores[1]*100:.2f}%")


model.save("modelo_cadencia_lstm_federado_manual_2.h5")
with open("labels_cadencia_lstm.json", "w") as f:
    json.dump(label_map, f)

