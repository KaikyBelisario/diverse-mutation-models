"""
treino_mutacao.py — Mutation Model (Khalifa et al., 2022 §3)
------------------------------------------------------------
CNN que aprende a imitar mutações bem-sucedidas do CPA (Assisted ou Normal).

Input : janela 8×8 recortada ao redor do ponto de mutação
        (narrow representation — PCGRL §2.3, Khalifa et al. §4.3)
Output: classificação em 4 ações
          0 = NoChange
          1 = Floor
          2 = Wall
          3 = Resource

Normal   : treina uma vez no EHR final (train_mutation_model com epochs=8).
Assisted : treina a cada N passos do CPA e re-injeta o modelo  (epochs=2).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from device_utils import get_device
from path_utils import ensure_results_dir, get_dataset_path, get_mutation_model_path
from domain import crop_around, FLOOR, WALL, RESOURCE, BASE, CROP_SIZE

# ── Mapeamento tile → classe de ação ─────────────────────────────────────────
N_ACTIONS = 4   # { NoChange=0, Floor=1, Wall=2, Resource=3 }

_TILE_TO_ACTION = {
    FLOOR    : 1,
    WALL     : 2,
    RESOURCE : 3,
    BASE     : 0,   # Base nunca é mudada — trata como NoChange
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MutationDataset(Dataset):
    """
    Converte o EHR em pares (crop 8×8 normalizado, ação).

    Cada entrada do EHR contém:
      parent       — mapa antes da mutação
      loc_r, loc_c — posição da mutação
      action       — tile resultante (FLOOR=0, WALL=1, RESOURCE=2)
    """
    def __init__(self, ehr_data: list):
        crops   = []
        actions = []

        for item in ehr_data:
            parent = np.array(item['parent'], dtype=np.float32)
            r      = int(item['loc_r'])
            c      = int(item['loc_c'])
            tile   = int(item['action'])

            window = crop_around(parent, r, c, size=CROP_SIZE)
            window /= 3.0   # normaliza para [0, 1]

            act = _TILE_TO_ACTION.get(tile, 0)
            crops.append(window)
            actions.append(act)

        self.crops   = torch.tensor(np.array(crops), dtype=torch.float32).unsqueeze(1)
        self.actions = torch.tensor(np.array(actions), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx):
        return self.crops[idx], self.actions[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Arquitetura CNN  — estilo DQN (Khalifa et al. §4.3)
# ─────────────────────────────────────────────────────────────────────────────

class MutationCNN(nn.Module):
    """
    CNN de 3 conv layers + 2 FC layers (estilo Atari DQN).

    Input : [B, 1, 8, 8]   — crop normalizado
    Output: [B, N_ACTIONS] — logits
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                               # 8→4
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                               # 4→2
        )
        # 128 * 2 * 2 = 512
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# Loop de Treinamento
# ─────────────────────────────────────────────────────────────────────────────

def train_mutation_model(
        ehr_data: list | None = None,
        epochs:   int         = 8,
        return_model: bool    = False,
) -> "nn.Module | None":
    """
    Treina o MutationCNN no EHR.

    Parâmetros
    ----------
    ehr_data     : lista de dicts do EHR (se None, carrega do disco)
    epochs       : nº de épocas
                   • 8  para o método Normal  (treino único no EHR final)
                   • 2  para o método Assisted (re-treino periódico durante CPA)
    return_model : se True, retorna o modelo treinado (usado no Assisted)
    """
    ensure_results_dir()
    device = get_device()

    if ehr_data is None:
        try:
            ehr_data = np.load(get_dataset_path(), allow_pickle=True).tolist()
        except FileNotFoundError:
            print(f"[MutationCNN] Erro: '{get_dataset_path()}' não encontrado. "
                  "Execute o Passo 1 primeiro!")
            return None

    print(f"[MutationCNN] Construindo dataset de {len(ehr_data):,} transições...")
    full_ds = MutationDataset(ehr_data)

    n       = len(full_ds)
    n_train = int(0.85 * n)
    n_val   = n - n_train

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    # Re-inicializa do zero a cada chamada (Khalifa et al. §4.3)
    model     = MutationCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for crops, actions in train_loader:
            crops, actions = crops.to(device), actions.to(device)
            optimizer.zero_grad()
            loss = criterion(model(crops), actions)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for crops, actions in val_loader:
                crops, actions = crops.to(device), actions.to(device)
                correct += (model(crops).argmax(1) == actions).sum().item()
                total   += actions.size(0)

        acc = correct / total if total > 0 else 0.0
        print(f"   Época [{epoch+1}/{epochs}] | "
              f"Loss: {t_loss/len(train_loader):.4f} | Val Acc: {acc*100:.1f}%")

    torch.save(model.state_dict(), get_mutation_model_path())
    print(f"[MutationCNN] Modelo salvo em '{get_mutation_model_path()}'.")

    # Libera optimizer e cache de GPU imediatamente (evita fragmentação de VRAM)
    del optimizer, criterion
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if return_model:
        model.eval()
        return model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Callback para Assisted Evolution
# ─────────────────────────────────────────────────────────────────────────────

def assisted_retrain_callback(ehr_data: list) -> nn.Module:
    """
    Treina o modelo com 2 épocas no EHR atual e retorna o modelo treinado.
    Passado como retrain_callback ao FICPA para Assisted Evolution.
    """
    print(f"[Assisted] Retreinando com {len(ehr_data):,} amostras (2 épocas)...")
    return train_mutation_model(ehr_data=ehr_data, epochs=2, return_model=True)


if __name__ == "__main__":
    train_mutation_model(epochs=8)
