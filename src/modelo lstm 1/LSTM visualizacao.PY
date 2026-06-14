import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # Usaremos para o cálculo eficiente da moda

# Configurações de Pastas
DATASET_ORIGINAL = "spectrograms_original"
PASTA_SAIDA = "visualizacoes"
os.makedirs(PASTA_SAIDA, exist_ok=True)

if not os.path.exists(DATASET_ORIGINAL):
    raise FileNotFoundError(f"A pasta '{DATASET_ORIGINAL}' não foi encontrada.")

# 1. Carregamento dos Dados
classes = sorted(os.listdir(DATASET_ORIGINAL))
dados_por_classe = {}

for classe in classes:
    caminho_classe = os.path.join(DATASET_ORIGINAL, classe)
    if not os.path.isdir(caminho_classe): continue
    
    espectrogramas = []
    for arquivo in os.listdir(caminho_classe):
        if arquivo.endswith(".npy"):
            img = np.load(os.path.join(caminho_classe, arquivo))
            espectrogramas.append(img)
            
    if espectrogramas:
        dados_por_classe[classe] = np.array(espectrogramas)

print(f"Processando as assinaturas funcionais para: {list(dados_por_classe.keys())}")

# =========================================================================
# VISUALIZAÇÃO: ENVELOPES COM 90 INTERVALOS, MÉDIA, MEDIANA E MODA
# =========================================================================
num_segmentos = 90

for classe, matriz_dados in dados_por_classe.items():
    # Extrai a energia temporal média das frequências (Amostras, 94)
    energia_temporal = np.mean(matriz_dados, axis=1)
    colunas_tempo = energia_temporal.shape[1]
    
    # Define os limites de corte para os 90 intervalos
    limites = np.linspace(0, colunas_tempo, num_segmentos + 1, dtype=int)
    
    minimos_seg = []
    maximos_seg = []
    medias_seg = []
    medianas_seg = []
    modas_seg = []
    eixo_x_valido = []
    
    for i in range(num_segmentos):
        start_idx = limites[i]
        end_idx = limites[i+1]
        
        if start_idx == end_idx: continue
        bloco = energia_temporal[:, start_idx:end_idx]
        if bloco.size == 0: continue
            
        # 1. Métricas de Limite e Centro Tradicionais
        minimos_seg.append(np.min(bloco))
        maximos_seg.append(np.max(bloco))
        medias_seg.append(np.mean(bloco))
        medianas_seg.append(np.median(bloco))
        
        # 2. Cálculo Matemático da Moda para Sinais Contínuos
        # Arredondamos para 2 casas decimais para agrupar valores float próximos em "baldes"
        bloco_discreto = np.round(bloco, decimals=2)
        valores_moda, contagens = stats.mode(bloco_discreto, axis=None, keepdims=False)
        modas_seg.append(valores_moda)
        
        eixo_x_valido.append(i)
        
    # --- PLOTAGEM DA NOVA VISUALIZAÇÃO COMPLETA ---
    plt.figure(figsize=(12, 6))
    
    # Envelope de amplitude de ruído (Área Cinza)
    plt.fill_between(eixo_x_valido, minimos_seg, maximos_seg, color='gray', alpha=0.15, label='Dispersão Total (Mín/Máx)')
    
    # Plot das 3 Funções Centrais para análise comparativa
    plt.plot(eixo_x_valido, medias_seg, color='blue', linewidth=2, label='Função Média')
    plt.plot(eixo_x_valido, medianas_seg, color='darkorange', linewidth=2, linestyle='-', label='Função Mediana')
    plt.plot(eixo_x_valido, modas_seg, color='purple', linewidth=1.8, linestyle=':', label='Função Moda (Ajustada)')
    
    # Linhas de limites externos
    plt.plot(eixo_x_valido, minimos_seg, color='red', linestyle='--', alpha=0.5, label='Limite Mínimo')
    plt.plot(eixo_x_valido, maximos_seg, color='green', linestyle='--', alpha=0.5, label='Limite Máximo')
    
    plt.title(f"Análise Estatística Avançada da Classe: '{classe.upper()}' ({num_segmentos} Intervalos)")
    plt.xlabel("Segmentos de Tempo de Alta Resolução")
    plt.ylabel("Intensidade Espectral")
    
    # Como são 90 intervalos, mostramos os marcadores no eixo X de 5 em 5 para ficar legível
    plt.xticks(np.arange(0, num_segmentos, 5))
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    # Salvando o arquivo
    caminho_salvamento = os.path.join(PASTA_SAIDA, f"assinatura_avancada_90seg_{classe}.png")
    plt.savefig(caminho_salvamento, bbox_inches='tight', dpi=150)
    plt.close()

print(f"Concluído! Verifique os novos gráficos de 90 intervalos na pasta: '{os.path.abspath(PASTA_SAIDA)}'")