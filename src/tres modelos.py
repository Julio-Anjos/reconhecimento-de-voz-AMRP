import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, LSTM, TimeDistributed, Permute

# =====================================================
# CONFIGURAÇÕES E DIRETÓRIOS
# =====================================================
PASTA_SAIDA = "comparacao_modelos_ruido_cadencia"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Dimensões exatas baseadas no seu pipeline (128 bandas Mel x 94 frames de tempo)
INPUT_SHAPE = (128, 94, 1) 

# 📌 Certifique-se de carregar aqui o seu dataset QUE CONTÉM RUÍDO.
# X_train shape esperado: (num_amostras, 128, 94, 1)
# Y_train shape esperado: (num_amostras, 5) -> One-Hot Encoded (ou use Categorical Crossentropy)
X_train, Y_train = None, None  
X_val, Y_val = None, None      
NUM_CLASSES = 5  # Configurado para as suas 5 classes (pare, siga, direita, esquerda, voltar)

EPOCHS = 35
BATCH_SIZE = 32

# =====================================================
# DEFINIÇÃO DAS ARQUITETURAS (FOCO EM RUÍDO E CADÊNCIA)
# =====================================================

def criar_modelo_ruido_equilibrado_50_50():
    """Equilíbrio: Convoluções limpam frequências ruidosas isoladas, mas 
    preservam a linha temporal (94 frames) intacta para o bloco LSTM."""
    model = Sequential([
        # Convolução com kernel (5, 1) atua essencialmente no eixo das frequências
        Conv2D(32, (5, 1), activation='relu', padding='same', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 1)), # Reduz apenas a resolução espectral, preserva o tempo
        Dropout(0.2),
        
        Conv2D(64, (5, 1), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),
        Dropout(0.2),
        
        # Alinha as dimensões para colocar o Tempo (94) como eixo sequencial primário
        Permute((2, 1, 3)), 
        TimeDistributed(Flatten()),
        
        # Bloco recorrente intermediário
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ], name="Modelo_Ruido_Equilibrado_50_50")
    return model


def criar_modelo_ruido_foco_lstm_70_30():
    """CNN atua levemente como um filtro de linha inicial. A análise baseia-se 
    a 70% nas curvas, acelerações e ritmo da fala gerados pelas camadas recorrentes."""
    model = Sequential([
        # Filtro de suavização geométrica básico
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 1)), 
        
        # Transposição para focar a sequência nos blocos temporais restantes
        Permute((2, 1, 3)),
        TimeDistributed(Flatten()),
        
        # LSTM Robusta: Duas camadas empilhadas para mapear a cadência silábica
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ], name="Modelo_Ruido_Foco_LSTM_70_30")
    return model


def criar_modelo_ruido_puro_lstm():
    """Sem convolução. Desfaz a ideia de imagem e trata as 128 bandas Mel 
    como 128 curvas de um gráfico que evoluem em simultâneo ao longo do tempo."""
    model = Sequential([
        # Remove a dimensão extra de canal e transpõe: (128, 94, 1) -> (94, 128)
        Reshape((128, 94), input_shape=INPUT_SHAPE),
        Permute((2, 1)), 
        
        # Camadas LSTM puras processando a cadência pura e ignorando ruídos estáticos continuos
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ], name="Modelo_Ruido_Puro_LSTM")
    return model

# =====================================================
# FUNÇÃO DE SALVAMENTO DE GRÁFICOS
# =====================================================
def salvar_visualizacoes(history, nome_modelo):
    """Gera e salva os gráficos de histórico na pasta de destino."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de Acurácia
    ax1.plot(history.history['accuracy'], label='Treino', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validação', color='orange')
    ax1.set_title(f'Acurácia no Ruído - {nome_modelo}')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de Perda (Loss)
    ax2.plot(history.history['loss'], label='Treino', color='blue')
    ax2.plot(history.history['val_loss'], label='Validação', color='orange')
    ax2.set_title(f'Perda (Loss) no Ruído - {nome_modelo}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    ax2.grid(True)
    
    caminho_grafico = os.path.join(PASTA_SAIDA, f"historico_{nome_modelo}.png")
    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=150)
    plt.close()
    print(f"📊 Gráficos salvos em: {caminho_grafico}")

# =====================================================
# LOOP DE TREINAMENTO E COMPARAÇÃO
# =====================================================

# Certificação de segurança para o caso de se esquecer de alimentar as variáveis
if X_train is None or Y_train is None:
    print("⚠️ ATENÇÃO: Carregue os seus arrays de dados com ruído no topo do script para iniciar o treino real.")
    # Arrays fictícios apenas para validação e compilação de teste do script
    X_train = np.random.randn(100, 128, 94, 1).astype(np.float32)
    Y_train = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, size=100), NUM_CLASSES)
    X_val, Y_val = X_train, Y_train

dicionario_modelos = {
    "ruido_50_50": criar_modelo_ruido_equilibrado_50_50(),
    "ruido_70_30": criar_modelo_ruido_foco_lstm_70_30(),
    "ruido_puro_lstm": criar_modelo_ruido_puro_lstm()
}

resultados_resumo = {}

for chave, modelo in dicionario_modelos.items():
    print("\n" + "="*60)
    print(f" Iniciando Treinamento: {modelo.name}")
    print("="*60)
    
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Exibir a nova estrutura interna de dimensões temporais no terminal
    modelo.summary()
    
    # Treinamento
    history = modelo.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    melhor_val_acc = max(history.history['val_accuracy'])
    resultados_resumo[modelo.name] = {
        "final_accuracy": history.history['accuracy'][-1],
        "best_val_accuracy": melhor_val_acc,
        "final_loss": history.history['loss'][-1]
    }
    
    # Guardar os pesos e arquitetura do novo modelo (.h5)
    caminho_modelo = os.path.join(PASTA_SAIDA, f"{modelo.name}.h5")
    modelo.save(caminho_modelo)
    print(f"Modelo de Cadência salvo com sucesso em: {caminho_modelo}")
    
    # Gerar gráficos específicos para esta versão temporal
    salvar_visualizacoes(history, modelo.name)

# Salvar ficheiro JSON comparativo na nova pasta
caminho_json = os.path.join(PASTA_SAIDA, "relatorio_comparativo_ruido.json")
with open(caminho_json, 'w') as f:
    json.dump(resultados_resumo, f, indent=4)

print("\n" + "="*60)
print("  PROCESSO DE TREINO CONCLUÍDO COM SUCESSO!")
print(f" Aceda à pasta '{PASTA_SAIDA}' para auditar as curvas de treino.")
print("="*60)