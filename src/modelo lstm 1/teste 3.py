import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. MAPEAMENTO DE CONFIGURAÇÕES
# ==========================================
DATASET_PATH = "dataset_final"
SAMPLE_RATE = 16000
DURATION = 1  # 1 segundo fixo para padronizar as matrizes
N_MFCC = 13
MAX_PAD_LEN = 32  # Número fixo de frames no tempo

classes = sorted(os.listdir(DATASET_PATH))
num_classes = len(classes)
print(f"Classes encontradas: {classes}")

# ==========================================
# 2. FUNÇÃO DE EXTRAÇÃO DE FEATURES (MFCC + DELTAS)
# ==========================================
def extract_features(file_path):
    # Carrega o áudio forçando a taxa de amostragem padrão
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Padroniza o tamanho do áudio para exatamente 1 segundo
    max_samples = SAMPLE_RATE * DURATION
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    else:
        audio = audio[:max_samples]
        
    # Extrai o MFCC básico
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    
    # Calcula Delta e Delta-Delta (Velocidade e Aceleração do som)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Empilha horizontalmente (Stack) para criar uma representação rica
    # Resultado terá formato: (39, frames_de_tempo)
    features = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
    
    # Transpõe para o formato que a LSTM espera: (frames_de_tempo, features)
    return features.T

# ==========================================
# 3. PREPARAÇÃO DO DATASET
# ==========================================
X, y = [], []
for label_idx, class_name in enumerate(classes):
    class_dir = os.path.join(DATASET_PATH, class_name)
    file_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    
    print(f"Processando {class_name}: {len(file_paths)} arquivos...")
    for path in file_paths:
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(label_idx)
        except Exception as e:
            print(f"Erro ao processar {path}: {e}")

X = np.array(X) # Formato: (num_audios, tempo, 39)
y = np.array(y)

# Divide em Treino e Validação (80% treino, 20% validação)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criando DataLoaders do PyTorch
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(AudioDataset(X_val, y_val), batch_size=16, shuffle=False)

# ==========================================
# 4. ARQUITETURA DO MODELO (LSTM BIDIRECIONAL)
# ==========================================
class AudioBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Bidirecional para capturar contexto do início ao fim e vice-versa
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0.0)
        
        # Como é bidirecional, a saída tem tamanho hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x formato: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        
        # Pegamos apenas o último passo no tempo (last hidden state) para classificar
        out = out[:, -1, :] 
        
        # Passa pela camada linear para dar os scores das 6 classes
        out = self.fc(out)
        return out

# Inicializa o modelo (39 features de entrada: 13 MFCC + 13 Delta + 13 Delta2)
model = AudioBiLSTM(input_size=39, hidden_size=64, num_layers=2, num_classes=num_classes)

# ==========================================
# 5. LOOP DE TREINAMENTO
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nIniciando o Treinamento...")
epochs = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validação simples
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = (correct / total) * 100
    print(f"Época [{epoch+1}/{epochs}] | Loss Treino: {train_loss/len(train_loader):.4f} | Acurácia Val: {acc:.2f}%")

# ==========================================
# 6. SALVANDO O MODELO E OS METADADOS
# ==========================================
# Salva os pesos do modelo estruturado
torch.save(model.state_state_dict(), "modelo_bilstm_audios.pth")

# Salva as classes em ordem correspondente para você usar na hora de predizer
import json
with open("classes.json", "w") as f:
    json.dump(classes, f)

print("\nModelo e mapeamento de classes salvos com sucesso!")