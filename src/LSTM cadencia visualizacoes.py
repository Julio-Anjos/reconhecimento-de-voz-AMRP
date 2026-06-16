import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 


DATASET_ORIGINAL = "spectrograms_original"
PASTA_SAIDA = "visualizacoes_cadencia"
os.makedirs(PASTA_SAIDA, exist_ok=True)

if not os.path.exists(DATASET_ORIGINAL):
    raise FileNotFoundError(f"A pasta '{DATASET_ORIGINAL}' não foi encontrada.")

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


num_segmentos = 90

for classe, matriz_dados in dados_por_classe.items():

    cadencia_temporal = np.sqrt(np.mean(matriz_dados ** 2, axis=1))
    colunas_tempo = cadencia_temporal.shape[1]
    
    
    limites = np.linspace(0, colunas_tempo, num_segmentos + 1, dtype=int)
    
    minimos_seg = []
    maximos_seg = []
    medias_seg = []
    medianas_seg = []
    modas_seg = []
    desvios_seg = []    
    variancias_seg = [] 
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
        
   
    medias_seg = np.array(medias_seg)
    desvios_seg = np.array(desvios_seg)
  
    plt.figure(figsize=(14, 7))
    
   
    plt.fill_between(eixo_x_valido, minimos_seg, maximos_seg, color='gray', alpha=0.1, 
                     label='Dispersão Absoluta (Mín/Máx)')
    

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

    print(f"\n Métricas de Afastamento da Média para a Classe [{classe.upper()}]:")
    print(f"   -> Variância Média Global : {np.mean(variancias_seg):.4f}")
    print(f"   -> Desvio Padrão Máximo   : {np.max(desvios_seg):.4f} (Maior oscilação de ritmo detectada no segmento {np.argmax(desvios_seg)})")
    print(f"   -> Desvio Padrão Mínimo   : {np.min(desvios_seg):.4f} (Momento de maior consistência temporal no segmento {np.argmin(desvios_seg)})")

print(f"\nProcesso concluído! Os gráficos focados em cadência e variância foram salvos em: '{os.path.abspath(PASTA_SAIDA)}'")