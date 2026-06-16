
import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Reshape, LSTM

# Criar diretório para salvar as imagens da apresentação
OUTPUT_DIR = "fluxo_arquitetura_ufc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. CARREGAR MODELO E LABELS
model = load_model("modelo_cadencia_lstm_federado.h5")
with open("labels_cadencia_lstm.json", "r") as f:
    label_map = json.load(f)
labels = list(label_map.keys())

SAMPLE_RATE = 16000
DURATION = 1
TARGET_SIZE = 64
rp = RecurrencePlot(threshold='point', percentage=20)

def process_single_audio_detailed(file_path):
    """Carrega o áudio e extrai os estados intermediários para o fluxograma"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    max_samples = SAMPLE_RATE * DURATION
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    else:
        audio = audio[:max_samples]
        
    # Estado Intermediário 1: Downsampling manual/nativo para 2kHz
    audio_resized = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=2000)
    
    
    matriz_distancia = np.abs(audio_resized[:, None] - audio_resized[None, :])
    
    # Estado Final: Matriz binarizada e interpolada para a CNN
    X_rp = rp.fit_transform(audio_resized.reshape(1, -1))
    scale_factor = TARGET_SIZE / X_rp.shape[1]
    matrix_cadence = zoom(X_rp[0], scale_factor, order=1)
    
    return audio, audio_resized, matriz_distancia, matrix_cadence

# 2. ENCONTRAR AS CAMADAS CORRETAS DINAMICAMENTE
model.build((None, TARGET_SIZE, TARGET_SIZE, 1))
camadas_conv = [layer for layer in model.layers if isinstance(layer, Conv2D)]
camada_reshape = [layer for layer in model.layers if isinstance(layer, Reshape)][0]
camada_lstm = [layer for layer in model.layers if isinstance(layer, LSTM)][0]

visualizador_cnn = Model(inputs=model.inputs, outputs=camadas_conv[-1].output)
visualizador_reshape = Model(inputs=model.inputs, outputs=camada_reshape.output)
visualizador_lstm = Model(inputs=model.inputs, outputs=camada_lstm.output)

# 3. BUSCAR UM ÁUDIO VÁLIDO DE CADA CLASSE
AUDIO_ORIGINAL = "dataset_final"
exemplos_por_classe = {}

for label in labels:
    path = os.path.join(AUDIO_ORIGINAL, label)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".wav"):
                exemplos_por_classe[label] = os.path.join(path, file)
                break

# 4. RASTREAMENTO COMPLETO DO FLUXO
for label, file_path in exemplos_por_classe.items():
    print(f"Gerando Painel de Fluxo -> Comando: {label.upper()}")
    
    # Captura todas as transições de sinais do projeto
    audio_bruto, audio_2k, matriz_dist, matriz_cadencia = process_single_audio_detailed(file_path)
    
    input_tensor = np.expand_dims(np.expand_dims(matriz_cadencia, axis=-1), axis=0)
    
    feat_cnn = visualizador_cnn.predict(input_tensor, verbose=0)
    feat_reshape = visualizador_reshape.predict(input_tensor, verbose=0)
    feat_lstm = visualizador_lstm.predict(input_tensor, verbose=0)
    pred_final = model.predict(input_tensor, verbose=0)[0]
    
    # =========================================================================
    # RESTRUTURAÇÃO DO PAINEL VISUAL (GRID 2x4 - PADRÃO ARTIGO CIENTÍFICO)
    # =========================================================================
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Fluxograma de Sinais e Ativações de IA - Comando: '{label.upper()}'", fontsize=16, fontweight='bold', y=0.98)
    
    # --- LINHA 1: ENGENHARIA DE RECURSOS (PROCESSAMENTO DO SINAL) ---
    
    # A. Áudio Bruto Capturado (16kHz)
    axs[0, 0].plot(np.linspace(0, 1, len(audio_bruto)), audio_bruto, color='#2c3e50', alpha=0.8)
    axs[0, 0].set_title("1. Onda de Áudio Original (1D)\n[Taxa: 16 kHz | Entrada Bruta]", fontsize=10, fontweight='bold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # B. Sinal com Downsampling (2kHz)
    axs[0, 1].plot(np.linspace(0, 1, len(audio_2k)), audio_2k, color='#16a085', alpha=0.9)
    axs[0, 1].set_title("2. Sinal Reduzido (Downsampling)\n[Taxa: 2 kHz | Prevenção de Memória]", fontsize=10, fontweight='bold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # C. Mapa de Distâncias Absolutas (Contínuo)
    axs[0, 2].imshow(matriz_dist, cmap='YlOrBr', origin='lower')
    axs[0, 2].set_title("3. Relevo de Fase Não-Linear\n[Matriz de Distância Absoluta]", fontsize=10, fontweight='bold')
    
    # D. Matriz de Cadência Binarizada (Entrada da Rede)
    axs[0, 3].imshow(matriz_cadencia, cmap='viridis', origin='lower')
    axs[0, 3].set_title(f"4. Gráfico de Recorrência (CNN Input)\n[Matriz de Cadência Binarizada {matriz_cadencia.shape}]", fontsize=10, fontweight='bold')
    
    # --- LINHA 2: FLUXO REDE NEURAL DE APRENDIZADO DE MÁQUINA ---
    
    # E. Filtros Espaciais da CNN
    media_filtros_cnn = np.mean(feat_cnn[0], axis=-1)
    axs[1, 0].imshow(media_filtros_cnn, cmap='inferno', origin='lower')
    axs[1, 0].set_title(f"5. Mapa Geométrico da CNN\n[Extração de Bordas de Ritmo]", fontsize=10, fontweight='bold')
    
    # F. Estruturação Cronológica para LSTM
    matriz_reshape = feat_reshape[0]
    axs[1, 1].imshow(matriz_reshape, cmap='coolwarm', aspect='auto')
    axs[1, 1].set_title("6. Vetor Temporalizado (Reshape)\n[1024 Recursos x 16 Passos]", fontsize=10, fontweight='bold')
    axs[1, 1].set_xlabel("Características da Imagem")
    axs[1, 1].set_ylabel("Cronologia (Tempo)")
    
    # G. A Série Temporal (Visão Estilo Bolsa de Valores)
    variancias = np.var(matriz_reshape, axis=0)
    indice_max_var = np.argmax(variancias)
    serie_temporal = matriz_reshape[:, indice_max_var]
    axs[1, 2].plot(range(16), serie_temporal, marker='o', color='#e74c3c', linewidth=2.5)
    axs[1, 2].set_title(f"7. Dinâmica de Maior Variância\n[Série Temporal do Recurso #{indice_max_var}]", fontsize=10, fontweight='bold')
    axs[1, 2].set_xlabel("Passos de Tempo (LSTM)")
    axs[1, 2].grid(True, linestyle='--', alpha=0.5)
    
    # H. Decisão Final (Softmax) + Assinatura de Memória
    axs[1, 3].bar(labels, pred_final, color='#27ae60', edgecolor='black')
    axs[1, 3].set_title("8. Probabilidades Finais (Softmax)\n[Classificação dos Nós Federados]", fontsize=10, fontweight='bold')
    axs[1, 3].set_ylim(0, 1.05)
    axs[1, 3].grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(axs[1, 3].get_xticklabels(), rotation=30, ha="right")
    
    # Mini-gráfico interno: Memória Oculta (Hidden State)
    ax_inset = axs[1, 3].inset_axes([0.52, 0.55, 0.45, 0.18])
    ax_inset.imshow(feat_lstm, cmap='magma', aspect='auto')
    ax_inset.set_title("Memória LSTM (64)", fontsize=8, fontweight='bold')
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    
    # Otimização de espaço e salvamento
    plt.tight_layout()
    nome_grafico = f"{OUTPUT_DIR}/fluxograma_completo_{label}.png"
    plt.savefig(nome_grafico, dpi=180)
    plt.show()
    print(f"Painel salvo com sucesso em: {nome_grafico}\n" + "="*50)