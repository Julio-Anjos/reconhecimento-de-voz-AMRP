# import os
# import sys
# import json
# import numpy as np
# import sounddevice as sd
# import keyboard
# from tensorflow.keras.models import load_model

# # Importando as funções de padronização que você usou no treino
# from utils import preprocess_audio, stft_manual, mel_filterbank

# # =====================================================
# # CONFIGURAÇÕES DA CAPTURA E MODELO NOVO
# # =====================================================
# SAMPLE_RATE = 16000    # Taxa de amostragem imposta pelo treino
# DURATION = 1.2         # Duração do áudio em segundos
# TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
# MAX_LEN = 94           # Dimensão temporal máxima do espectrograma

# # 📌 ALTERE AQUI PARA OS NOMES DO SEU MODELO NOVO:
# MODEL_PATH = "modelo_mic_simulacao1.h5"  # Coloque o nome do novo .h5 ou .keras
# LABELS_PATH = "labels_mic_simulacao1.json"

# # Limites de decisão (Filtros contra falso-positivo)
# THRESHOLD = 0.5
# ENTROPY_LIMIT = 1.4

# # =====================================================
# # PIPELINE DE ÁUDIO E PADRONIZAÇÃO
# # =====================================================

# def gravar_audio():
#     """Captura o áudio do microfone nos moldes do treinamento."""
#     # Gravação mono (channels=1) em float32
#     audio = sd.rec(int(DURATION * SAMPLE_RATE),
#                    samplerate=SAMPLE_RATE,
#                    channels=1,
#                    dtype='float32')
#     sd.wait()  # Aguarda o término dos 1.2 segundos de gravação
#     return audio.flatten()


# def audio_para_mel(audio):
#     """Transforma o áudio bruto no formato exato que o modelo espera."""
#     # 1. Pré-processamento (garante o tamanho fixo do vetor de áudio)
#     audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

#     # 2. Extração de características espectrais (STFT)
#     spec = stft_manual(audio)
#     spec_power = np.abs(spec) ** 2

#     # 3. Aplicação do Banco de Filtros Mel (128 bandas / FFT de 512)
#     mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)
#     mel_spec = np.dot(mel_fb, spec_power.T)

#     # 4. Conversão para a escala Log/Decibéis (com tratamento contra log de 0)
#     mel_db = 10 * np.log10(mel_spec + 1e-10)

#     # 5. Normalização Z-score (essencial para tirar ruído estático do microfone)
#     mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

#     # 6. Ajuste dimensional rigoroso no tempo (MAX_LEN)
#     if mel_db.shape[1] < MAX_LEN:
#         mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])))
#     else:
#         mel_db = mel_db[:, :MAX_LEN]

#     return mel_db


# def calcular_entropia(probs):
#     """Calcula a incerteza do modelo (quanto menor, mais seguro ele está)."""
#     probs = probs + 1e-10
#     return -np.sum(probs * np.log(probs))

# # =====================================================
# # CARREGAMENTO DO MODELO NOVO
# # =====================================================
# print("Carregando o novo modelo e dicionário de classes...")
# if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
#     print(f"❌ ERRO: Verifique se os arquivos '{MODEL_PATH}' e '{LABELS_PATH}' estão na pasta!")
#     sys.exit()

# model = load_model(MODEL_PATH)

# with open(LABELS_PATH) as f:
#     label_map = json.load(f)

# # Inverte o mapa de labels {0: "comando1", 1: "comando2"}
# inv_map = {int(v): k for k, v in label_map.items()}

# # =====================================================
# # MOTOR DE CLASSIFICAÇÃO
# # =====================================================
# def classificar():
#     print("\n🎤 Gravando... fale agora!")
#     audio = gravar_audio()
#     print("🧠 Processando sinal e inferindo...")

#     # Extrai as features nos moldes do treino
#     mel = audio_para_mel(audio)

#     # Adiciona as dimensões extras de Batch e Channel que o Keras exige:
#     # Formato final esperado: (1, 128, 94, 1)
#     mel = np.expand_dims(mel, axis=-1)
#     mel = np.expand_dims(mel, axis=0)

#     # Realiza a predição com o modelo novo
#     pred = model.predict(mel, verbose=0)[0]

#     idx = np.argmax(pred)
#     conf = pred[idx]
#     ent = calcular_entropia(pred)

#     classe = inv_map[idx]

#     # Filtros de Segurança contra barulhos genéricos ou incertezas da IA
#     if classe.lower() == "ruido" or conf < THRESHOLD or ent > ENTROPY_LIMIT:
#         classe = "desconhecido/inseguro"

#     print("-" * 40)
#     print(f" Cobaias / Resultado: {classe.upper()}")
#     print(f" Confiança do Modelo: {conf * 100:.2f}%")
#     print(f" Entropia Espacial  : {ent:.4f}")
#     print("-" * 40)


# def loop():
#     print("\n" + "="*50)
#     print(" SISTEMA DE RECONHECIMENTO DE VOZ INTERATIVO")
#     print(" Pressione [ G ] para Gravar um comando")
#     print(" Pressione [ Q ] para Sair do programa")
#     print("="*50 + "\n")

#     while True:
#         try:
#             if keyboard.is_pressed('q'):
#                 print("Saindo...")
#                 break
#             if keyboard.is_pressed('g'):
#                 classificar()
#                 # Pequena pausa para evitar que um clique duplo dispare duas gravações
#                 sd.sleep(500) 
#                 print("\nPronto para o próximo comando! (G = Gravar | Q = Sair)")
#         except Exception as e:
#             print(f"Erro no loop: {e}")
#             break

# # Iniciar aplicação
# loop()

import os
import sys
import json
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
from pyts.image import RecurrencePlot
from scipy.ndimage import zoom


SAMPLE_RATE = 16000    
DURATION = 1.2         
TARGET_SIZE = 64       

# Modelos Federados
MODEL_V1_PATH = "modelo_cadencia_lstm_federado.h5"         
MODEL_V2_PATH = "modelo_cadencia_lstm_federado2.h5"       
LABELS_PATH = "labels_cadencia_lstm.json"


rp = RecurrencePlot(threshold='point', percentage=20)


def pipeline_v1_recorrencia(audio_bruto):
    """Transforma o áudio em uma Matriz de Recorrência Estática (1 Canal)."""
   
    intervalos = librosa.effects.split(audio_bruto, top_db=20)
    if len(intervalos) > 0:
        audio_cortado = audio_bruto[intervalos[0][0]:intervalos[-1][1]]
    else:
        audio_cortado = audio_bruto


    target_samples = int(SAMPLE_RATE * DURATION)
    if len(audio_cortado) < target_samples:
        audio_final = np.pad(audio_cortado, (0, target_samples - len(audio_cortado)), 'constant')
    else:
        audio_final = audio_cortado[:target_samples]
        
    # Normalização Simples Z-Score
    audio_norm = (audio_final - np.mean(audio_final)) / (np.std(audio_final) + 1e-6)
    
    # Downsampling para viabilizar o cálculo da matriz quadrada
    audio_resized = librosa.resample(audio_norm, orig_sr=SAMPLE_RATE, target_sr=2000)
    
    # Geração da Matriz de Recorrência (Relações não-lineares de distância)
    X_rp = rp.fit_transform(audio_resized.reshape(1, -1))[0]
    
    # Força dimensão espacial 64x64 via interpolação bilinear
    scale_factor = TARGET_SIZE / X_rp.shape[0]
    matriz_v1 = zoom(X_rp, scale_factor, order=1)
    
    # Adiciona dimensão de canal (1 canal - Escala de cinza) -> Shape: (64, 64, 1)
    return np.expand_dims(matriz_v1, axis=-1)


def pipeline_v2_cadencia_3canais(audio_bruto):
    """Aplica VAD Suave, Centralização Estrita e extrai 3 canais de velocidade/aceleração."""
    # VAD Suave (top_db=35) para preservar consoantes de baixa energia como 'S' e 'F'
    intervalos = librosa.effects.split(audio_bruto, top_db=35)
    if len(intervalos) > 0:
        audio_fala = audio_bruto[intervalos[0][0]:intervalos[-1][1]]
    else:
        audio_fala = audio_bruto


    target_samples = int(SAMPLE_RATE * 1.0)
    if len(audio_fala) < target_samples:
        total_pad = target_samples - len(audio_fala)
        pad_esquerdo = total_pad // 2
        pad_direito = total_pad - pad_esquerdo
        audio_final = np.pad(audio_fala, (pad_esquerdo, pad_direito), 'constant')
    else:
        start_crop = (len(audio_fala) - target_samples) // 2
        audio_final = audio_fala[start_crop:start_crop + target_samples]
    
    pico = np.max(np.abs(audio_final))
    if pico > 1e-6:
        audio_final = audio_final / pico

  
    mel_spec = librosa.feature.melspectrogram(y=audio_final, sr=SAMPLE_RATE, n_mels=64, n_fft=512, hop_length=256)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Extração Dinâmica da Cadência (Velocidade e Aceleração Temporal)
    delta_velocidade = librosa.feature.delta(mel_db, order=1)
    delta2_aceleracao = librosa.feature.delta(mel_db, order=2)
    
    # Ajuste de escala bilinear para conformidade com a entrada da CNN (64x64)
    scale_y = TARGET_SIZE / mel_db.shape[1]
    canal_1 = zoom(mel_db, (1, scale_y), order=1)
    canal_2 = zoom(delta_velocidade, (1, scale_y), order=1)
    canal_3 = zoom(delta2_aceleracao, (1, scale_y), order=1)
    
    # Empilhamento Psuedo-RGB -> Shape: (64, 64, 3)
    return np.stack([canal_1, canal_2, canal_3], axis=-1)



print("Carregando Modelos Federados V1 e V2...")
if not os.path.exists(MODEL_V1_PATH) or not os.path.exists(MODEL_V2_PATH) or not os.path.exists(LABELS_PATH):
    print("ERRO: Verifique se os arquivos dos modelos e labels estão no diretório atual.")
    sys.exit()

model_v1 = load_model(MODEL_V1_PATH)
model_v2 = load_model(MODEL_V2_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)
inv_map = {int(v): k for k, v in label_map.items()}

def executar_comparativo():
    print("\nCapturando áudio (1.2s)... Fale agora!")
    audio_bruto = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio_bruto = audio_bruto.flatten()
    
    print("Processando informações em paralelo através dos dois pipelines...")
    
    # Execução das Transformações Específicas
    input_v1 = pipeline_v1_recorrencia(audio_bruto)
    input_v2 = pipeline_v2_cadencia_3canais(audio_bruto)
    
    # Inserção da dimensão de lote (Batch dimension) exigida pelo Keras
    batch_v1 = np.expand_dims(input_v1, axis=0) # Formato final: (1, 64, 64, 1)
    batch_v2 = np.expand_dims(input_v2, axis=0) # Formato final: (1, 64, 64, 3)
    
    # Predições Universais
    pred_v1 = model_v1.predict(batch_v1, verbose=0)[0]
    pred_v2 = model_v2.predict(batch_v2, verbose=0)[0]
    
    idx_v1 = np.argmax(pred_v1)
    idx_v2 = np.argmax(pred_v2)
    
    print("\n" + "="*60)
    print(" RESULTADOS COMPARATIVOS DE TRATAMENTO DE SINAL")
    print("="*60)
    print(f" [MODELO V1 - RECORRÊNCIA ESTÁTICA]:")
    print(f"   -> Estrutura dos Dados Gerada : {input_v1.shape} (Matriz Geométrica 1 Canal)")
    print(f"   -> Comando Predito            : {inv_map[idx_v1].upper()}")
    print(f"   -> Confiança de Predição      : {pred_v1[idx_v1]*100:.2f}%")
    print("-" * 60)
    print(f" [MODELO V2 - CADÊNCIA LINEAR EM 3 CANAIS]:")
    print(f"   -> Estrutura dos Dados Gerada : {input_v2.shape} (Espectrograma + Δ + Δ² em 3 Canais)")
    print(f"   -> Comando Predito            : {inv_map[idx_v2].upper()}")
    print(f"   -> Confiança de Predição      : {pred_v2[idx_v2]*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    executar_comparativo()