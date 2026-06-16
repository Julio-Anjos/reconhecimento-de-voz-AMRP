import os
import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, LSTM, Dense, Dropout
)
from tensorflow.keras.losses import CategoricalCrossentropy
from scipy.ndimage import zoom


DATASET_FINAL = "dataset_final"

SAMPLE_RATE = 16000
DURATION = 1.0  
TARGET_SIZE = 64 

if not os.path.exists(DATASET_FINAL):
    raise FileNotFoundError(f"ERRO: A pasta '{DATASET_FINAL}' não foi encontrada.")

labels = sorted(os.listdir(DATASET_FINAL))
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)



def isolar_e_centralizar_fala(audio, top_db=30):
    """
    [ESTRATÉGIA 1] Remove o silêncio inútil e centraliza a palavra falada 
    perfeitamente no meio da janela temporal, eliminando os espaços vazios enganosos.
    """
    intervalos = librosa.effects.split(audio, top_db=top_db)
    if len(intervalos) > 0:
        # Extrai apenas o trecho onde há energia de voz
        audio_fala = audio[intervalos[0][0]:intervalos[-1][1]]
    else:
        audio_fala = audio

    target_samples = int(SAMPLE_RATE * DURATION)
    
  
    if len(audio_fala) < target_samples:
        total_pad = target_samples - len(audio_fala)
        pad_esquerdo = total_pad // 2
        pad_direito = total_pad - pad_esquerdo
        audio_final = np.pad(audio_fala, (pad_esquerdo, pad_direito), 'constant')
    else:
       
        start_crop = (len(audio_fala) - target_samples) // 2
        audio_final = audio_fala[start_crop:start_crop + target_samples]
        
    return audio_final


def injetar_ruido_matematico(audio, pasta_dataset_base, ganho_min=0.03, ganho_max=0.15):
    """Data Augmentation focado em robustez contra névoas acústicas."""
    caminho_classe_ruido = os.path.join(pasta_dataset_base, "ruido")
    if not os.path.exists(caminho_classe_ruido): return audio 
    arquivos_ruido = [f for f in os.listdir(caminho_classe_ruido) if f.endswith('.wav')]
    if len(arquivos_ruido) == 0: return audio
        
    ruido_selecionado = np.random.choice(arquivos_ruido)
    audio_ruido, _ = librosa.load(os.path.join(caminho_classe_ruido, ruido_selecionado), sr=SAMPLE_RATE)
    
    if len(audio_ruido) < len(audio):
        audio_ruido = np.pad(audio_ruido, (0, len(audio) - len(audio_ruido)), 'wrap')
    else:
        audio_ruido = audio_ruido[:len(audio)]
        
    fator_ganho = np.random.uniform(ganho_min, ganho_max)
    return audio + (audio_ruido * fator_ganho)


def extrair_recursos_cadencia_3canais(file_path, aplicar_augmentation=False):
    """
    [ESTRATÉGIA 2] Gera um Espectrograma de 3 canais (Estático, Velocidade/Delta e Aceleração/Delta-Delta).
    Isso força a rede a aprender de forma linear o ritmo e a cadência da fala.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    audio = isolar_e_centralizar_fala(audio)
    
    pico = np.max(np.abs(audio))
    if pico > 1e-6: audio = audio / pico
        
    if aplicar_augmentation and "ruido" not in file_path.lower():
        audio = injetar_ruido_matematico(audio, DATASET_FINAL)
        pico = np.max(np.abs(audio))
        if pico > 1e-6: audio = audio / pico


    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=TARGET_SIZE, n_fft=512, hop_length=256)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    delta_velocidade = librosa.feature.delta(mel_db, order=1)
    # Delta-Delta captura a aceleração e explosão de sílabas
    delta2_aceleracao = librosa.feature.delta(mel_db, order=2)
    
    
    scale_y = TARGET_SIZE / mel_db.shape[1]
    
    canal_1 = zoom(mel_db, (1, scale_y), order=1)
    canal_2 = zoom(delta_velocidade, (1, scale_y), order=1)
    canal_3 = zoom(delta2_aceleracao, (1, scale_y), order=1)
    

    espectrograma_3d = np.stack([canal_1, canal_2, canal_3], axis=-1)
    return espectrograma_3d


X_dados, y_dados = [], []

print("Extraindo recursos lineares de Cadência (3 Canais)...")
for label in labels:
    path = os.path.join(DATASET_FINAL, label)
    if not os.path.isdir(path): continue

    print(f" -> Analisando ritmo da classe: {label}")
    for file in os.listdir(path):
        if file.endswith(".wav"):
            caminho_completo = os.path.join(path, file)
            try:
                matrix = extrair_recursos_cadencia_3canais(caminho_completo, aplicar_augmentation=True)
                X_dados.append(matrix)
                y_dados.append(label_map[label])
            except Exception as e:
                print(f"Falha no arquivo {file}: {e}")

X_dados = np.array(X_dados)
y_dados = np.array(y_dados)

X_train, X_val, y_train, y_val = train_test_split(
    X_dados, y_dados, test_size=0.2, stratify=y_dados, random_state=42
)

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

model.save("modelo_cadencia_lstm_federado2.h5")
with open("labels_cadencia_lstm.json", "w") as f:
    json.dump(label_map, f)

print("\n🏁 Modelo Atualizado com Sucesso!")