import os
import sys
import json
import numpy as np
import sounddevice as sd
import keyboard
from tensorflow.keras.models import load_model

# Importando as funções de padronização que você usou no treino
from utils import preprocess_audio, stft_manual, mel_filterbank

# =====================================================
# CONFIGURAÇÕES DA CAPTURA E MODELO NOVO
# =====================================================
SAMPLE_RATE = 16000    # Taxa de amostragem imposta pelo treino
DURATION = 1.2         # Duração do áudio em segundos
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
MAX_LEN = 94           # Dimensão temporal máxima do espectrograma

# 📌 ALTERE AQUI PARA OS NOMES DO SEU MODELO NOVO:
MODEL_PATH = "./modelo_cadencia_lstm_federado_manual_2.h5"  
LABELS_PATH = "./modelo_cadencia.json"

# Limites de decisão (Filtros contra falso-positivo)
THRESHOLD = 0.5
ENTROPY_LIMIT = 1.4

# =====================================================
# PIPELINE DE ÁUDIO E PADRONIZAÇÃO
# =====================================================

def gravar_audio():
    """Captura o áudio do microfone nos moldes do treinamento."""
    # Gravação mono (channels=1) em float32
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()  # Aguarda o término dos 1.2 segundos de gravação
    return audio.flatten()


def audio_para_mel(audio):
    """Transforma o áudio bruto no formato exato que o modelo espera."""
    # 1. Pré-processamento (garante o tamanho fixo do vetor de áudio)
    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

    # 2. Extração de características espectrais (STFT)
    spec = stft_manual(audio)
    spec_power = np.abs(spec) ** 2

    # 3. Aplicação do Banco de Filtros Mel (128 bandas / FFT de 512)
    mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)
    mel_spec = np.dot(mel_fb, spec_power.T)

    # 4. Conversão para a escala Log/Decibéis (com tratamento contra log de 0)
    mel_db = 10 * np.log10(mel_spec + 1e-10)

    # 5. Normalização Z-score (essencial para tirar ruído estático do microfone)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # 6. Ajuste dimensional rigoroso no tempo (MAX_LEN)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db


def calcular_entropia(probs):
    """Calcula a incerteza do modelo (quanto menor, mais seguro ele está)."""
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))

# =====================================================
# CARREGAMENTO DO MODELO NOVO
# =====================================================
print("Carregando o novo modelo e dicionário de classes...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print(f"❌ ERRO: Verifique se os arquivos '{MODEL_PATH}' e '{LABELS_PATH}' estão na pasta!")
    sys.exit()

model = load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)

# Inverte o mapa de labels {0: "comando1", 1: "comando2"}
inv_map = {int(v): k for k, v in label_map.items()}

# =====================================================
# MOTOR DE CLASSIFICAÇÃO
# =====================================================
def classificar():
    print("\n🎤 Gravando... fale agora!")
    audio = gravar_audio()
    print("🧠 Processando sinal e inferindo...")

    # Extrai as features nos moldes do treino
    mel = audio_para_mel(audio)

    # Adiciona as dimensões extras de Batch e Channel que o Keras exige:
    # Formato final esperado: (1, 128, 94, 1)
    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    # Realiza a predição com o modelo novo
    pred = model.predict(mel, verbose=0)[0]

    idx = np.argmax(pred)
    conf = pred[idx]
    ent = calcular_entropia(pred)

    classe = inv_map[idx]

    # Filtros de Segurança contra barulhos genéricos ou incertezas da IA
    if classe.lower() == "ruido" or conf < THRESHOLD or ent > ENTROPY_LIMIT:
        classe = "desconhecido/inseguro"

    print("-" * 40)
    print(f" Cobaias / Resultado: {classe.upper()}")
    print(f" Confiança do Modelo: {conf * 100:.2f}%")
    print(f" Entropia Espacial  : {ent:.4f}")
    print("-" * 40)


def loop():
    print("\n" + "="*50)
    print(" SISTEMA DE RECONHECIMENTO DE VOZ INTERATIVO")
    print(" Pressione [ G ] para Gravar um comando")
    print(" Pressione [ Q ] para Sair do programa")
    print("="*50 + "\n")

    while True:
        try:
            if keyboard.is_pressed('q'):
                print("Saindo...")
                break
            if keyboard.is_pressed('g'):
                classificar()
                # Pequena pausa para evitar que um clique duplo dispare duas gravações
                sd.sleep(500) 
                print("\nPronto para o próximo comando! (G = Gravar | Q = Sair)")
        except Exception as e:
            print(f"Erro no loop: {e}")
            break

# Iniciar aplicação
loop()
