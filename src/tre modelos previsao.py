import os
import sys
import json
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model

# Importando o seu pipeline padrão do arquivo utils
from utils import preprocess_audio, stft_manual, mel_filterbank


PASTA_MODELOS = "comparacao_modelos_ruido_cadencia"  
LABELS_PATH = "labels_mic_simulacao1.json"

CLASSES = ["pare", "siga", "direita", "esquerda", "voltar"]
REPETICOES_TOTAIS = 5  

SAMPLE_RATE = 16000
DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
MAX_LEN = 94

def gravar_audio():
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def audio_para_mel(audio):
    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)
    spec = stft_manual(audio)
    spec_power = np.abs(spec) ** 2
    mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)
    mel_spec = np.dot(mel_fb, spec_power.T)
    mel_db = 10 * np.log10(mel_spec + 1e-10)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db

def calcular_entropia(probs):
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))

# =====================================================
# CARREGAMENTO DOS MODELOS DE CADÊNCIA
# =====================================================
if not os.path.exists(LABELS_PATH):
    print(f"ERRO: O arquivo '{LABELS_PATH}' precisa estar nesta pasta!")
    sys.exit()

with open(LABELS_PATH) as f:
    label_map = json.load(f)
inv_map = {int(v): k for k, v in label_map.items()}

modelos_carregados = {}
nomes_modelos = ["Modelo_Ruido_Equilibrado_50_50", "Modelo_Ruido_Foco_LSTM_70_30", "Modelo_Ruido_Puro_LSTM"]

print("\nCarregando os modelos baseados em Cadência e Ruído...")
for nome in nomes_modelos:
    caminho = os.path.join(PASTA_MODELOS, f"{nome}.h5")
    if os.path.exists(caminho):
        modelos_carregados[nome] = load_model(caminho)
        print(f" -> {nome} carregado com sucesso!")
    else:
        print(f"AVISO: {caminho} não encontrado.")

if not modelos_carregados:
    print(" ERRO: Nenhum modelo novo de cadência foi localizado.")
    sys.exit()

# Métricas para o relatório final
estatisticas = {
    nome: {"acertos": 0, "confiancas_corretas": [], "entropias": []} 
    for nome in modelos_carregados.keys()
}
total_testes_global = len(CLASSES) * REPETICOES_TOTAIS

print("\n" + "="*60)
print("  SESSÃO DE PREVISÃO TRIPLA: FOCO EM RITMO E CADÊNCIA")
print("="*60)

for rodada in range(REPETICOES_TOTAIS):
    print(f"\n --- INICIANDO CICLO {rodada + 1}/{REPETICOES_TOTAIS} ---")
    
    for classe_real in CLASSES:
        print(f"\nPALAVRA ALVO AGORA: [ {classe_real.upper()} ]")
        input("Pressione [ ENTER ] e fale a palavra de forma natural...")
        
        print("Gravando cadência...")
        audio = gravar_audio()
        print("Áudio capturado. Fazendo previsão tripla...")
        
        # Base de características Mel (128, 94)
        mel_base = audio_para_mel(audio)
        
        print(f"\nResultados para o comando real '{classe_real}':")
        
        for nome_modelo, modelo in modelos_carregados.items():

            if "Puro_LSTM" in nome_modelo:
                # O Puro LSTM redesenha a matriz sem canais de imagem: shape (1, 128, 94)
                mel_input = np.expand_dims(mel_base, axis=0)
            else:
                # Os modelos híbridos (50_50 e 70_30) exigem o canal da CNN: shape (1, 128, 94, 1)
                mel_input = np.expand_dims(mel_base, axis=-1)
                mel_input = np.expand_dims(mel_input, axis=0)
            
            # Inferência imediata
            pred = modelo.predict(mel_input, verbose=0)[0]
            idx_pred = np.argmax(pred)
            classe_pred = inv_map[idx_pred]
            conf = pred[idx_pred]
            ent = calcular_entropia(pred)
            
            estatisticas[nome_modelo]["entropias"].append(ent)
            
            if classe_pred.lower().strip() == classe_real.lower().strip():
                status = "ACERTOU!"
                estatisticas[nome_modelo]["acertos"] += 1
                estatisticas[nome_modelo]["confiancas_corretas"].append(conf)
            else:
                status = f"ERROU (Identificou: [{classe_pred.upper()}])"
                
            print(f"  ▪️ {nome_modelo:<32}: {status} | Certeza: {conf*100:2.1f}% | Entropia: {ent:.3f}")
        print("-" * 60)


print("\n" + "="*75)
print(" BALANÇO DE PERFORMANCE DOS MODELOS DE CADÊNCIA COM RUÍDO")
print("="*75)

for nome_modelo, dados in estatisticas.items():
    acuracia = (dados["acertos"] / total_testes_global) * 100
    conf_media = np.mean(dados["confiancas_corretas"]) * 100 if dados["confiancas_corretas"] else 0.0
    estabilidade = np.std(dados["entropias"]) if dados["entropias"] else float('inf')
    
    print(f"{nome_modelo.upper()}:")
    print(f"   -> Taxa de Orientação Correta : {dados['acertos']}/{total_testes_global} ({acuracia:.1f}%)")
    print(f"   -> Confiança Média nos Acertos: {conf_media:.1f}%")
    print(f"   -> Instabilidade com Ruído    : {estabilidade:.4f} (Valores menores = Modelo mais firme)")
    print("-" * 60)