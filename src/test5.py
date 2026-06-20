import os
import json
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model

# Importando suas funções do utils.py
from utils import preprocess_audio, stft_manual, mel_filterbank

# --- CONFIGURAÇÕES DE ÁUDIO ---
# SAMPLE_RATE = 16000
# DURATION = 1.2
# TARGET_LENGTH = int(SAMPLE_RATE * DURATION)  # 19200 amostras
# MAX_LEN = 94

# # --- CONFIGURAÇÕES DO MODELO ---
# # Altere para o modelo que deseja testar:
# # "modelo_mic_simulacao1.h5" ou "modelo_cadencia_lstm_federado_manual.h5"
# MODEL_PATH = "modelo_cadencia_lstm_federado_manual_2.h5" 
# LABELS_PATH = "labels_cadencia_lstm.json"
# --- CONFIGURAÇÕES DE ÁUDIO ---
SAMPLE_RATE = 16000
DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)  # 19200 amostras

# MODIFIQUE AQUI: Mude de 94 para 32 para alinhar com o novo modelo LSTM
MAX_LEN = 32 

# --- CONFIGURAÇÕES DO MODELO ---
MODEL_PATH = "modelo_cadencia_lstm_federado_manual_2.h5" 
LABELS_PATH = "labels_cadencia_lstm.json"

THRESHOLD = 0.6          # Subi um pouco para evitar falsos positivos no tempo real
ENTROPY_LIMIT = 1.2      # Limite de incerteza do modelo

# --- BUFFER CIRCULAR PARA TEMPO REAL ---
# Este buffer armazena continuamente as últimas amostras do microfone
audio_buffer = np.zeros(TARGET_LENGTH, dtype=np.float32)

def audio_callback(indata, frames, time, status):
    """Esta função é chamada automaticamente pelo sounddevice a cada novo bloco de áudio do mic"""
    global audio_buffer
    if status:
        print(status)
    
    # Rotaciona o buffer: joga os dados antigos para trás e insere os novos no fim
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

def audio_para_mel(audio):
    # 1. Pré-processamento manual do seu utils
    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

    # 2. Extração de Features (STFT -> Mel Filterbank -> dB)
    spec = stft_manual(audio)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)
    mel_spec = np.dot(mel_fb, spec_power.T)

    # Evita divisão por zero ou log de zero
    mel_db = 10 * np.log10(mel_spec + 1e-10)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # 3. Ajuste do tamanho da matriz (Eixo do Tempo)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db

def calcular_entropia(probs):
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))

# --- CARREGAMENTO DO MODELO ---
print(f"📦 Carregando modelo {MODEL_PATH}...")
model = load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}

# --- LOOP PRINCIPAL EM TEMPO REAL ---
def iniciar_tempo_real():
    print("\n🎤 Ouvindo microfone em tempo real...")
    print("Pressione CTRL+C no terminal para encerrar.\n")
    
    # Define o tamanho do bloco de captura (ex: 2048 amostras ~128ms de latência)
    block_size = 2048 
    
    # Configura e abre o stream de entrada do microfone
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=block_size,
        dtype='float32'
    )
    
    ultima_predicao = ""

    with stream:
        while True:
            # Copia o estado atual do buffer para evitar problemas de concorrência
            janela_audio = np.copy(audio_buffer)
            
            # Se o buffer ainda estiver majoritariamente vazio (silêncio inicial), espera preencher
            if np.max(np.abs(janela_audio)) < 0.01:
                sd.sleep(100)
                continue

            # Processa o áudio atual do buffer
            mel = audio_para_mel(janela_audio)
            
            # Ajusta o shape para a rede ([Batch, Altura, Largura, Canais])
            mel = np.expand_dims(mel, axis=-1)
            mel = np.expand_dims(mel, axis=0)

            # Predição do modelo
            pred = model.predict(mel, verbose=0)[0]
            idx = np.argmax(pred)
            conf = pred[idx]
            ent = calcular_entropia(pred)

            classe = inv_map[idx]

            # Regras de rejeição de ruído/incerteza
            if classe == "ruido" or conf < THRESHOLD or ent > ENTROPY_LIMIT:
                classe = "desconhecido"

            # Só exibe no terminal se a classe mudar, evitando inundar a tela com "desconhecido"
            if classe != "desconhecido" and classe != ultima_predicao:
                print(f"🗣️ Comando Detectado: [ {classe.upper()} ] | Confiança: {conf:.2f} | Entropia: {ent:.2f}")
                ultima_predicao = classe
            elif classe == "desconhecido":
                ultima_predicao = ""

            # Pequena pausa milimétrica para não estressar a CPU (analisa o áudio ~10 vezes por segundo)
            sd.sleep(100)

if __name__ == "__main__":
    try:
        iniciar_tempo_real()
    except KeyboardInterrupt:
        print("\n🛑 Monitoramento encerrado pelo usuário.")