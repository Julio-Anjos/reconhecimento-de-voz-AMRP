import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, LSTM, TimeDistributed, Permute


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

PASTA_SAIDA = "comparacao_modelos_ruido_cadencia"
os.makedirs(PASTA_SAIDA, exist_ok=True)

INPUT_SHAPE = (128, 94, 1) 


X_train, Y_train = None, None  
X_val, Y_val = None, None      
NUM_CLASSES = 5  # Configurado para as suas 5 classes (pare, siga, direita, esquerda, voltar)
NOME_CLASSES = ["pare", "siga", "direita", "esquerda", "voltar"]

EPOCHS = 35
BATCH_SIZE = 32




def criar_modelo_ruido_equilibrado_50_50():
    """Equilíbrio: Convoluções limpam frequências ruidosas isoladas, mas 
    preservam a linha temporal (94 frames) intacta para o bloco LSTM."""
    model = Sequential([
        Conv2D(32, (5, 1), activation='relu', padding='same', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 1)), 
        Dropout(0.2),
        
        Conv2D(64, (5, 1), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),
        Dropout(0.2),
        
        Permute((2, 1, 3)), 
        TimeDistributed(Flatten()),
        
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
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 1)), 
        
        Permute((2, 1, 3)),
        TimeDistributed(Flatten()),
        
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
        Reshape((128, 94), input_shape=INPUT_SHAPE),
        Permute((2, 1)), 
        
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ], name="Modelo_Ruido_Puro_LSTM")
    return model

def salvar_visualizacoes(history, nome_modelo):
    """Gera e salva os gráficos de histórico na pasta de destino."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
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
    print(f"Gráficos de histórico salvos em: {caminho_grafico}")



# Certificação de segurança para o caso de se esquecer de alimentar as variáveis
if X_train is None or Y_train is None:
    print("ATENÇÃO: Carregue os seus arrays de dados com ruído no topo do script para iniciar o treino real.")
    
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
    
   
    modelo.summary()
    
    # Treinamento
    history = modelo.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Guardar os pesos e arquitetura do modelo (.h5)
    caminho_modelo = os.path.join(PASTA_SAIDA, f"{modelo.name}.h5")
    modelo.save(caminho_modelo)
    print(f"Modelo salvo com sucesso em: {caminho_modelo}")
    
    # Gerar gráficos de evolução (Acurácia e Loss ao longo das épocas)
    salvar_visualizacoes(history, modelo.name)
    
  
    print(f"\nEvaluating e extraindo métricas detalhadas para: {modelo.name}...")
    
    
    val_loss, val_accuracy = modelo.evaluate(X_val, Y_val, verbose=0)
    
    y_pred_softmax = modelo.predict(X_val)
    y_pred_classes = np.argmax(y_pred_softmax, axis=1) # Converte probabilidades para índices (0 a 4)
    y_true_classes = np.argmax(Y_val, axis=1)          # Converte One-Hot Real para índices (0 a 4)
    
  
    relatorio_dict = classification_report(
        y_true_classes, y_pred_classes, 
        target_names=NOME_CLASSES, output_dict=True
    )
    
    # Exibir o relatório completo no terminal de forma legível
    print("\nRELATÓRIO DE CLASSIFICAÇÃO POR CLASSE:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=NOME_CLASSES))
    
  
    matriz_confusao = confusion_matrix(y_true_classes, y_pred_classes)
    
  
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz_confusao, annot=True, fmt='d', cmap='Blues',
        xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES
    )
    plt.title(f'Matriz de Confusão (Absoluta) - {modelo.name}')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    caminho_matriz = os.path.join(PASTA_SAIDA, f"matriz_confusao_{modelo.name}.png")
    plt.tight_layout()
    plt.savefig(caminho_matriz, dpi=150)
    plt.close()
    print(f"🖼️ Gráfico de Matriz de Confusão salvo em: {caminho_matriz}")
    
   
    precisao_plot = [relatorio_dict[cls]['precision'] for cls in NOME_CLASSES]
    recall_plot = [relatorio_dict[cls]['recall'] for cls in NOME_CLASSES]
    
    x = np.arange(len(NOME_CLASSES))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, precisao_plot, width, label='Precisão', color='skyblue')
    ax.bar(x + width/2, recall_plot, width, label='Recall (Sensibilidade)', color='lightcoral')
    
    ax.set_ylabel('Pontuação (0 a 1)')
    ax.set_title(f'Métricas por Classe - {modelo.name}')
    ax.set_xticks(x)
    ax.set_xticklabels(NOME_CLASSES)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    caminho_metricas_barra = os.path.join(PASTA_SAIDA, f"metricas_por_classe_{modelo.name}.png")
    plt.tight_layout()
    plt.savefig(caminho_metricas_barra, dpi=150)
    plt.close()
    print(f" Gráfico de barras de métricas salvo em: {caminho_metricas_barra}")

  
    resultados_resumo[modelo.name] = {
        "val_loss_final": float(val_loss),
        "val_accuracy_global": float(val_accuracy),
        "detalhes_por_classe": relatorio_dict
    }

caminho_json = os.path.join(PASTA_SAIDA, "relatorio_comparativo_ruido.json")

def default_serializer(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    raise TypeError

with open(caminho_json, 'w') as f:
    json.dump(resultados_resumo, f, indent=4, default=default_serializer)

print("\n" + "="*60)
print("PROCESSO DE TREINO E AVALIAÇÃO CONCLUÍDO COM SUCESSO!")
print(f" Verifique a pasta '{PASTA_SAIDA}' para auditar os resultados.")
print("="*60)