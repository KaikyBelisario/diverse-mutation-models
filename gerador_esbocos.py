import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from device_utils import get_device
from path_utils import ensure_results_dir, get_dataset_path, get_sketch_model_path

GRID_SIZE = 14

# --- 1. FUNÇÃO PARA EXTRAIR AS CARACTERÍSTICAS (IGUAL AO PASSO 1) ---
def calculate_features(level):
    """Recalcula BC1 e BC2 para usarmos como condições de treino."""
    np_level = np.array(level)
    bc1 = np.sum(np_level) / (GRID_SIZE * GRID_SIZE)
    bc2 = np.sum(np_level[:, :-1] != np_level[:, 1:]) / (GRID_SIZE * (GRID_SIZE - 1))
    return [bc1, bc2]

# --- 2. DATASET CONDICIONAL ---
class SketchDataset(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        
        conditions = []
        targets = []
        
        for item in data:
            child = item['child']
            # A condição de entrada são as características do nível desejado
            bc_vector = calculate_features(child)
            conditions.append(bc_vector)
            targets.append(child)
            
        # Condições: Shape [N, 2] (Densidade, Complexidade)
        self.conditions = torch.tensor(conditions, dtype=torch.float32)
        # Alvos: Shape [N, 1, 10, 10] (O layout do labirinto correspondente)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.conditions[idx], self.targets[idx]


# --- 3. ARQUITETURA DO GERADOR DE ESBOÇOS CONDICIONAL ---
class ConditionalSketchGenerator(nn.Module):
    """
    Rede que recebe um vetor de tamanho 2 [BC1, BC2] e expande
    até gerar uma matriz de probabilidade de tamanho 10x10.
    Versão aprimorada com maior capacidade e BatchNorm para treino estável.
    """
    def __init__(self):
        super(ConditionalSketchGenerator, self).__init__()

        # Etapa 1: Expansão do vetor de condições com mais capacidade
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Etapa 2: Deconvoluções para crescer de 4x4 para 10x10
        # 32 canais x 4x4 = 512 — compatível com a camada linear acima
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=0),  # 4x4 -> 9x9
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=0),   # 9x9 -> 10x10
            nn.Sigmoid()
        )

    def forward(self, condition):
        x = self.fc(condition)
        x = x.view(-1, 32, 4, 4)  # 512 = 32 canais * 4 * 4
        return self.deconv(x)

# --- 4. LOOP DE TREINAMENTO ---
def train_sketch_generator():
    ensure_results_dir()

    print("-> Preparando dados para o Gerador de Esboços...")
    try:
        dataset = SketchDataset(get_dataset_path())
    except FileNotFoundError:
        print(f"Erro: '{get_dataset_path()}' não encontrado. Execute o Passo 1!")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = get_device()
    model = ConditionalSketchGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCELoss() # Compara a matriz gerada com a real bloco a bloco
    
    epochs = 60
    print(f"-> Treinando o Gerador de Esboços em: {device}...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_conditions, batch_targets in dataloader:
            conditions = batch_conditions.to(device)
            targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(conditions)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Época [{epoch+1}/{epochs}] | Erro de Condicionamento (Loss): {avg_loss:.4f}")
            
    torch.save(model.state_dict(), get_sketch_model_path())
    print(f"-> Passo 3 Concluído! Gerador de Esboços salvo em '{get_sketch_model_path()}'.")

if __name__ == "__main__":
    train_sketch_generator()