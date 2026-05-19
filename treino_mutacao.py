import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from device_utils import get_device
from path_utils import ensure_results_dir, get_dataset_path, get_mutation_model_path

# --- 1. PREPARAÇÃO E ESTRUTURAÇÃO DOS DADOS ---
class MutationDataset(Dataset):
    """
    Dataset customizado para carregar os pares (Pai, Filho) gerados pelo MAP-Elites.
    Transforma matrizes normais em Tensores do PyTorch.
    """
    def __init__(self, filepath):
        # Carrega o arquivo gerado no Passo 1
        data = np.load(filepath, allow_pickle=True)
        
        # Extrai as listas de pais e filhos
        parents_raw = [item['parent'] for item in data]
        children_raw = [item['child'] for item in data]
        
        # Converte para float32 e adiciona uma dimensão de 'canal' (Shape final: [N, 1, 10, 10])
        # O PyTorch espera o formato: [Batch_Size, Canais, Altura, Largura]
        self.parents = torch.tensor(parents_raw, dtype=torch.float32).unsqueeze(1)
        self.children = torch.tensor(children_raw, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.parents)
        
    def __getitem__(self, idx):
        return self.parents[idx], self.children[idx]


# --- 2. ARQUITETURA DA REDE NEURAL (Residual Learning) ---
class MutationCNN(nn.Module):
    """
    CNN com aprendizado residual: em vez de reconstruir o labirinto inteiro,
    a rede aprende apenas o DELTA (diferença) a aplicar no mapa pai.
    Isso evita o colapso para a função identidade quando pai e filho são muito similares.
    """
    def __init__(self):
        super(MutationCNN, self).__init__()

        # Encoder: extrai características espaciais do mapa atual
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder: produz um delta (variação), não o mapa completo
        # Tanh gera valores entre -1 e +1, representando "remover" ou "adicionar" parede
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        delta = self.decoder(features)
        # Aplica pequenas modificações ao mapa original
        # O fator 0.5 controla a "agressividade" das mutações
        return torch.clamp(x + delta * 0.5, 0.0, 1.0)


# --- 3. LOOP DE TREINAMENTO ---
def train_model():
    ensure_results_dir()

    print("-> Carregando o dataset de mutações...")
    try:
        dataset = MutationDataset(get_dataset_path())
    except FileNotFoundError:
        print(f"Erro: O arquivo '{get_dataset_path()}' não foi encontrado. Execute o Passo 1 primeiro!")
        return

    # DataLoader divide os dados em pacotes (batches) e embaralha-os para um treino estável
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"-> Dataset carregado com {len(dataset)} transições evolutivas para treino.")

    # Inicializa a rede, o otimizador (Adam) e a função de perda (BCELoss)
    device = get_device()
    model = MutationCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    epochs = 60  # Quantas vezes a IA vai rever todo o dataset
    print(f"-> Iniciando treinamento em: {device}...")

    

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_parents, batch_children in dataloader:
            # Move os dados para a GPU (se disponível) ou CPU
            inputs = batch_parents.to(device)
            targets = batch_children.to(device)
            
            # Zera os gradientes do passo anterior
            optimizer.zero_grad()
            
            # 1. Forward pass: A IA tenta adivinhar o Filho com base no Pai
            outputs = model(inputs)
            
            # 2. Calcula o Erro (Loss) comparando com o Filho real do MAP-Elites
            loss = criterion(outputs, targets)
            
            # 3. Backward pass: Calcula o impacto de cada neurónio no erro
            loss.backward()
            
            # 4. Atualiza os pesos da rede para errar menos na próxima vez
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Época [{epoch+1}/{epochs}] | Erro Médio (Loss): {avg_loss:.4f}")

    # Salva os pesos treinados num ficheiro binário
    torch.save(model.state_dict(), get_mutation_model_path())
    print(f"-> Treinamento concluído! Modelo guardado em '{get_mutation_model_path()}'.")

if __name__ == "__main__":
    train_model()