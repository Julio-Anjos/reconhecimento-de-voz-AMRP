import os
import numpy as np
import matplotlib.pyplot as plt

# Configurações de Pastas
DATASET_ORIGINAL = "spectrograms_original" # Seus dados brutos ou arrays processados
PASTA_SAIDA = "visualizacoes_rp"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Se você tiver os áudios em formato .npy (ondas temporais), use esse script.
# Caso seus .npy já sejam espectrogramas, este script aplicará a recorrência sobre a matriz mel.

classes = sorted(os.listdir(DATASET_ORIGINAL))

# --- FUNÇÃO MATEMÁTICA VETORIZADA PARA GRAFICO DE RECORRÊNCIA ---
def calcular_recurrence_plot(sinal, dimensao_saida=150, eps=0.1):
    """
    Transforma um sinal temporal em uma matriz de recorrência bidimensional.
    Usa amostragem linear para fixar o tamanho da imagem final.
    """
    # Garante que o sinal seja estritamente 1D
    sinal_1d = sinal.flatten()
    
    # Downsampling para redimensionar o sinal para o tamanho da matriz desejada
    indices = np.linspace(0, len(sinal_1d) - 1, dimensao_saida, dtype=int)
    sinal_reduzido = sinal_1d[indices]
    
    # Truque geométrico do NumPy: calcula a distância euclidiana de todos contra todos
    # usando transmissão (broadcasting) de matrizes
    matriz_distancia = np.abs(sinal_reduzido[:, None] - sinal_reduzido[None, :])
    
    # Binarização: Se a distância for menor que o limiar (eps), há recorrência (1), senão (0)
    # Deixamos contínuo (sem binarizar) para a CNN extrair mais detalhes de textura,
    # invertendo para que pontos próximos fiquem escuros/claros de forma nítida.
    matriz_recorrencia = np.exp(-matriz_distancia / eps)
    
    return matriz_recorrencia

# --- PROCESSAMENTO EM LOTE ---
print("Iniciando a extração dos Gráficos de Recorrência...")

for classe in classes:
    caminho_classe = os.path.join(DATASET_ORIGINAL, classe)
    if not os.path.isdir(caminho_classe): continue
    
    arquivos = [f for f in os.listdir(caminho_classe) if f.endswith(".npy")]
    if not arquivos: continue
    
    print(f"-> Gerando RPs para a classe: '{classe.upper()}' (Processando 3 amostras visuais)")
    
    # Vamos gerar as 3 primeiras amostras de cada classe para você comparar visualmente
    for i, arquivo in enumerate(arquivos[:3]):
        dado = np.load(os.path.join(caminho_classe, arquivo))
        
        # Gera o mapa de recorrência
        rp_matrix = calcular_recurrence_plot(dado, dimensao_saida=150, eps=0.15)
        
        # Plotagem
        plt.figure(figsize=(6, 6))
        # Usamos cmap='binary' ou 'plasma' para expor as texturas geométricas
        plt.imshow(rp_matrix, cmap='plasma', origin='lower')
        plt.title(f"Recurrence Plot: {classe.upper()} (Amostra {i+1})")
        plt.xlabel("Tempo (t)")
        plt.ylabel("Tempo (s)")
        
        # Salvamento
        nome_saida = f"rp_{classe}_amostra_{i+1}.png"
        plt.savefig(os.path.join(PASTA_SAIDA, nome_saida), bbox_inches='tight', dpi=120)
        plt.close()

print(f"\n[SUCESSO] Gráficos salvos na pasta: '{os.path.abspath(PASTA_SAIDA)}'")
print("Abra as imagens e compare os padrões de textura entre as diferentes classes!")