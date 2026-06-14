import os
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Reshape, LSTM, Dense, Dropout, Lambda
)
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================================================================
# 1. CARREGAMENTO DOS DADOS (REPLICANDO SEU PIPELINE PADRÃO)
# =========================================================================

DATASET_ORIGINAL = "spectrograms_original"
DATASET_RUIDO = "spectrograms_ruido"

labels = sorted(os.listdir(DATASET_ORIGINAL))
label_map = {label: i for i, label in enumerate(labels)}

X_orig, y_orig = [], []
for label in labels:
    path = os.path.join(DATASET_ORIGINAL, label)
    if not os.path.isdir(path): continue
    for file in os.listdir(path):
        if file.endswith(".npy"):
            mel = np.load(os.path.join(path, file))
            mel = np.expand_dims(mel, axis=-1)
            X_orig.append(mel)
            y_orig.append(label_map[label])

X_orig, y_orig = np.array(X_orig), np.array(y_orig)

# Divisão de treino e validação para o áudio de referência (Limpo)
X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42
)

# Carregamento dos dados com Ruído (Aumentação)
X_aug, y_aug = [], []
for label in labels:
    path = os.path.join(DATASET_RUIDO, label)
    if not os.path.isdir(path): continue
    for file in os.listdir(path):
        if file.endswith(".npy"):
            mel = np.load(os.path.join(path, file))
            mel = np.expand_dims(mel, axis=-1)
            X_aug.append(mel)
            y_aug.append(label_map[label])

X_aug, y_aug = np.array(X_aug), np.array(y_aug)

# Balanceamento simples do ruído
MAX_AUG_PER_CLASS = 800
X_aug_bal, y_aug_bal = [], []
for label in range(len(labels)):
    idx = np.where(y_aug == label)[0]
    np.random.shuffle(idx)
    idx = idx[:MAX_AUG_PER_CLASS]
    X_aug_bal.append(X_aug[idx])
    y_aug_bal.append(y_aug[idx])

X_aug_bal = np.concatenate(X_aug_bal)
y_aug_bal = np.concatenate(y_aug_bal)

# =========================================================================
# 2. CRIAÇÃO DOS PARES COLABORATIVOS (Para treinar a Rede Siamesa)
# O modelo siamês precisa receber pares de imagens e uma etiqueta:
# 1 se forem o mesmo comando, 0 se forem comandos diferentes.
# =========================================================================
def criar_pares_colaborativos(X_limpo, y_limpo, X_ruido, y_ruido):
    pares_A, pares_B, labels_pares = [], [], []
    
    # Criando pares positivos (Mesmo comando: Áudio Limpo + Versão com Ruído)
    for i in range(len(X_limpo)):
        img_limpa = X_limpo[i]
        classe = y_limpo[i]
        
        # Acha amostras da mesma classe no dataset com ruído
        idx_mesma_classe = np.where(y_ruido == classe)[0]
        img_ruidosa = X_ruido[np.random.choice(idx_mesma_classe)]
        
        pares_A.append(img_limpa)
        pares_B.append(img_ruidosa)
        labels_pares.append(1.0) # 1 = Mesma função matemática/comando
        
    # Criando pares negativos (Comandos diferentes: Áudio Limpo + Ruído de Outro Comando)
    for i in range(len(X_limpo)):
        img_limpa = X_limpo[i]
        classe = y_limpo[i]
        
        # Acha amostras de classes diferentes no dataset com ruído
        idx_classe_diferente = np.where(y_ruido != classe)[0]
        img_ruidosa_errada = X_ruido[np.random.choice(idx_classe_diferente)]
        
        pares_A.append(img_limpa)
        pares_B.append(img_ruidosa_errada)
        labels_pares.append(0.0) # 0 = Funções matemáticas totalmente distintas
        
    return np.array(pares_A), np.array(pares_B), np.array(labels_pares)

print("Gerando pares colaborativos para treinamento...")
X_train_A, X_train_B, y_train_siames = criar_pares_colaborativos(X_train_orig, y_train_orig, X_aug_bal, y_aug_bal)
X_val_A, X_val_B, y_val_siames = criar_pares_colaborativos(X_val, y_val, X_val, y_val)

# =========================================================================
# 3. DEFINIÇÃO DA ARQUITETURA SIAMESA CRNN
# =========================================================================
input_shape = X_train_A.shape[1:]

def criar_extrator_crnn(shape):
    inputs = Input(shape=shape)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    tempo_reduzido = shape[1] // 4
    freq_reduzida = shape[0] // 4
    x = Reshape(target_shape=(tempo_reduzido, freq_reduzida * 64))(x)
    
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    embedding = Dense(32, activation='linear')(x) # Representação em função contínua
    return Model(inputs, embedding, name="Extrator_Base")

# Instancia o extrator compartilhado
extrator_compartilhado = criar_extrator_crnn(input_shape)

# Entradas paralelas da rede siamesa
input_A = Input(shape=input_shape)
input_B = Input(shape=input_shape)

funcao_A = extrator_compartilhado(input_A)
funcao_B = extrator_compartilhado(input_B)

# Comparação geométrica de distância
calcular_distancia = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1, keepdims=True)))
distancia_geometrica = calcular_distancia([funcao_A, funcao_B])

modelo_siames_completo = Model(inputs=[input_A, input_B], outputs=distancia_geometrica)

# Perda por Contraste (Contrastive Loss)
def contrastive_loss(y_true, distance):
    margin = 1.0
    square_pred = K.square(distance)
    margin_square = K.square(K.maximum(margin - distance, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

modelo_siames_completo.compile(optimizer='adam', loss=contrastive_loss)

# =========================================================================
# 4. TREINAMENTO E SALVAMENTO
# =========================================================================
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3)
]

print("Iniciando treinamento colaborativo...")
modelo_siames_completo.fit(
    [X_train_A, X_train_B], y_train_siames,
    validation_data=([X_val_A, X_val_B], y_val_siames),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# CRITICAL: Salvamos apenas o extrator base que gera o vetor/função descritiva.
# É esse arquivo que o seu arquivo de predição/inspeção unitária vai ler.
extrator_compartilhado.save("modelo_siames_base.keras")

with open("labels_mic_simulacao1.json", "w") as f:
    json.dump(label_map, f)

print("Mó dulos e assinaturas de funções salvas com absoluto sucesso!")