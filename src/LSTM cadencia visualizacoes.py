import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

# Configurações de Pastas
DATASET_ORIGINAL = "spectrograms_original"
PASTA_SAIDA = "visualizacoes_cadencia"
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

print(f"Processando as assinaturas funcionais de cadência para: {list(dados_por_classe.keys())}")

# =========================================================================
# VISUALIZAÇÃO: ENVELOPES DE CADÊNCIA, VARIÂNCIA E DESVIO PADRÃO
# =========================================================================
num_segmentos = 90

for classe, matriz_dados in dados_por_classe.items():
    # 🎙️ EXTRAÇÃO DA CADÊNCIA REAIS: Calculamos a raiz da média dos quadrados (RMS) 
    # ao longo do eixo das frequências para obter o envelope de amplitude temporal legítimo.
    # Formato resultante: (Amostras, 94)
    cadencia_temporal = np.sqrt(np.mean(matriz_dados ** 2, axis=1))
    colunas_tempo = cadencia_temporal.shape[1]
    
    # Define os limites de corte para os 90 intervalos
    limites = np.linspace(0, colunas_tempo, num_segmentos + 1, dtype=int)
    
    minimos_seg = []
    maximos_seg = []
    medias_seg = []
    medianas_seg = []
    modas_seg = []
    desvios_seg = []    # 📉 Desvio Padrão por segmento
    variancias_seg = [] # 📊 Variância por segmento
    eixo_x_valido = []
    
    for i in range(num_segmentos):
        start_idx = limites[i]
        end_idx = limites[i+1]
        
        if start_idx == end_idx: continue
        bloco = cadencia_temporal[:, start_idx:end_idx]
        if bloco.size == 0: continue
            
        # 1. Métricas Estatísticas Tradicionais
        minimos_seg.append(np.min(bloco))
        maximos_seg.append(np.max(bloco))
        medias_seg.append(np.mean(bloco))
        medianas_seg.append(np.median(bloco))
        
        # 2. Métricas de Afastamento da Média (Dispersão)
        desvios_seg.append(np.std(bloco))
        variancias_seg.append(np.var(bloco))
        
        # 3. Moda Ajustada
        bloco_discreto = np.round(bloco, decimals=2)
        valores_moda, contagens = stats.mode(bloco_discreto, axis=None, keepdims=False)
        modas_seg.append(valores_moda)
        
        eixo_x_valido.append(i)
        
    # Converte para arrays numpy para facilitar operações matemáticas vetoriais
    medias_seg = np.array(medias_seg)
    desvios_seg = np.array(desvios_seg)
    
    # --- PLOTAGEM DO GRÁFICO DE CADÊNCIA AVANÇADO ---
    plt.figure(figsize=(14, 7))
    
    # Envelope 1: Dispersão Absoluta Total (Mínimo ao Máximo) - Cinza Claro
    plt.fill_between(eixo_x_valido, minimos_seg, maximos_seg, color='gray', alpha=0.1, 
                     label='Dispersão Absoluta (Mín/Máx)')
    
    # Envelope 2: ZONA DE DESVIO PADRÃO (O quanto os áudios se distanciam da Média)
    # Mostra a região de 1 Desvio Padrão para cima e para baixo da Função Média
    plt.fill_between(eixo_x_valido, 
                     medias_seg - desvios_seg, 
                     medias_seg + desvios_seg, 
                     color='purple', alpha=0.2, label='Zona de Desvio Padrão ($\sigma$ ao redor da Média)')
    
    # Linhas Centrais
    plt.plot(eixo_x_valido, medias_seg, color='blue', linewidth=2.5, label='Função Média (Cadência)')
    plt.plot(eixo_x_valido, medianas_seg, color='darkorange', linewidth=2, linestyle='-', label='Função Mediana')
    plt.plot(eixo_x_valido, modas_seg, color='green', linewidth=1.5, linestyle=':', label='Função Moda')
    
    # Detalhes do Gráfico
    plt.title(f"Assinatura Funcional de Cadência e Ritmo - Classe: '{classe.upper()}'", fontsize=14, fontweight='bold')
    plt.xlabel("Segmentos de Tempo de Alta Resolução (Cadência)", fontsize=11)
    plt.ylabel("Intensidade da Energia Temporal (RMS)", fontsize=11)
    
    plt.xticks(np.arange(0, num_segmentos, 5))
    plt.grid(True, alpha=0.25, linestyle='--')
    plt.legend(loc='upper right')
    
    # Salvar Gráfico
    caminho_salvamento = os.path.join(PASTA_SAIDA, f"cadencia_dispersao_{classe}.png")
    plt.savefig(caminho_salvamento, bbox_inches='tight', dpi=150)
    plt.close()
    
    # --- RELATÓRIO IMPRESSO NO TERMINAL ---
    print(f"\n📈 Métricas de Afastamento da Média para a Classe [{classe.upper()}]:")
    print(f"   -> Variância Média Global : {np.mean(variancias_seg):.4f}")
    print(f"   -> Desvio Padrão Máximo   : {np.max(desvios_seg):.4f} (Maior oscilação de ritmo detectada no segmento {np.argmax(desvios_seg)})")
    print(f"   -> Desvio Padrão Mínimo   : {np.min(desvios_seg):.4f} (Momento de maior consistência temporal no segmento {np.argmin(desvios_seg)})")

print(f"\n🏁 Processo concluído! Os gráficos focados em cadência e variância foram salvos em: '{os.path.abspath(PASTA_SAIDA)}'")