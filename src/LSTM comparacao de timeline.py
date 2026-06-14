import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyts.image import RecurrencePlot
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. CONFIGURAÇÕES E CARREGAMENTO
# ==========================================
AUDIO_ORIGINAL = "dataset_final"
SAMPLE_RATE = 16000
DURATION = 1  
TARGET_SIZE = 64 

# Carregar dicionário de classes
with open("labels_cadencia_lstm.json", "r") as f:
    label_map = json.load(f)
labels = sorted(list(label_map.keys()))
num_classes = len(labels)

print("Carregando os 3 modelos salvos...")
model_antigo = load_model("modelo_mic_simulacao1.h5")      # CNN + Espectrograma
model_cnn_puro = load_model("modelo_cnn_puro_cadencia.h5")  # CNN + Cadência (Centralizado)
model_hibrido = load_model("modelo_cadencia_lstm_federado_manual.h5") # CRNN + Cadência (Federado)
 
rp = RecurrencePlot(threshold='point', percentage=20)

# ==========================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO DIAGNÓSTICO
# ==========================================
def extrair_espectrograma(audio):
    """Gera o espectrograma de Mel para o modelo antigo"""
    # Ajuste os parâmetros (n_mels, n_fft) caso seu modelo antigo use valores diferentes
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Redimensiona se necessário para bater com o input_shape do modelo_mic_simulacao1
    # Aqui assumimos que ele espera algo parecido com (64, TARGET_SIZE, 1)
    mel_resized = zoom(mel_spec_db, (TARGET_SIZE / mel_spec_db.shape[0], TARGET_SIZE / mel_spec_db.shape[1]), order=1)
    return np.expand_dims(mel_resized, axis=-1)

def extrair_cadencia(audio):
    """Gera a matriz de recorrência para os modelos novos"""
    audio_resized = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=2000)
    X_rp = rp.fit_transform(audio_resized.reshape(1, -1))
    scale_factor = TARGET_SIZE / X_rp.shape[1]
    matrix_cadence = zoom(X_rp[0], scale_factor, order=1)
    return np.expand_dims(matrix_cadence, axis=-1)

# ==========================================
# 3. EXTRAÇÃO DO CONJUNTO DE TESTE UNIFICADO
# ==========================================
X_spec_test, X_cad_test, y_test = [], [], []

print("\nCarregando dados de teste e gerando representações...")
for label in labels:
    path = os.path.join(AUDIO_ORIGINAL, label)
    if not os.path.isdir(path): continue
    
    arquivos = [f for f in os.listdir(path) if f.endswith(".wav")]
    # Usaremos os últimos 20% de arquivos de cada pasta para garantir dados não vistos no treino puro
    cut_off = int(len(arquivos) * 0.8)
    test_files = arquivos[cut_off:]
    
    for file in test_files:
        try:
            audio, sr = librosa.load(os.path.join(path, file), sr=SAMPLE_RATE)
            max_samples = SAMPLE_RATE * DURATION
            if len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
            else:
                audio = audio[:max_samples]
            
            # Gera as duas features para o mesmo arquivo de áudio
            X_spec_test.append(extrair_espectrograma(audio))
            X_cad_test.append(extrair_cadencia(audio))
            y_test.append(label_map[label])
        except:
            pass

X_spec_test = np.array(X_spec_test)
X_cad_test = np.array(X_cad_test)
y_test = np.array(y_test)

# ==========================================
# 4. INFERÊNCIA E PREDIÇÕES
# ==========================================
print("\nExecutando predições nos 3 modelos...")
preds_antigo = model_antigo.predict(X_spec_test, verbose=0)
preds_cnn_puro = model_cnn_puro.predict(X_cad_test, verbose=0)
preds_hibrido = model_hibrido.predict(X_cad_test, verbose=0)

acc_antigo = accuracy_score(y_test, np.argmax(preds_antigo, axis=1))
acc_cnn_puro = accuracy_score(y_test, np.argmax(preds_cnn_puro, axis=1))
acc_hibrido = accuracy_score(y_test, np.argmax(preds_hibrido, axis=1))

# ==========================================
# 5. GERAÇÃO DAS VISUALIZAÇÕES COMPARATIVAS
# ==========================================

# Visualização 1: Gráfico de Barras de Acurácia Global
plt.figure(figsize=(8, 5))
model_names = ['1. Espectrograma (Antigo)', '2. Cadência (CNN Puro)', '3. Cadência + LSTM (Federado)']
accuracies = [acc_antigo * 100, acc_cnn_puro * 100, acc_hibrido * 100]
bars = plt.bar(model_names, accuracies, color=['#95a5a6', '#3498db', '#2ecc71'], edgecolor='black', width=0.6)
plt.title("Evolução da Acurácia Global por Modelo", fontsize=12, fontweight='bold')
plt.ylabel("Acurácia (%)")
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.show()

# Visualização 2: Curva ROC Macro-Média Comparativa
plt.figure(figsize=(8, 6))
y_test_bin = label_binarize(y_test, classes=range(num_classes))

for preds, name, color in zip([preds_antigo, preds_cnn_puro, preds_hibrido], model_names, ['gray', 'blue', 'green']):
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), preds.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=color, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Comparação de Curvas ROC Globais', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*50)
print("RELATÓRIO CLÍNICO COMPARATIVO DE MELHORIAS:")
print(f"Acurácia Base (Espectrograma): {acc_antigo*100:.2f}%")
print(f"Impacto da Mudança de Feature (Cadência): {((acc_cnn_puro - acc_antigo)*100):+.2f}% de ganho.")
print(f"Impacto da Dinâmica Temporal + FedAvg: {((acc_hibrido - acc_cnn_puro)*100):+.2f}% de ganho.")
print(f"Melhoria Total Acumulada no Projeto: {((acc_hibrido - acc_antigo)*100):+.2f}% de ganho.")
print("="*50)