import torch
import torch.nn as nn
import numpy as np
from device_utils import get_device
from path_utils import get_sketch_model_path, get_mutation_model_path

# --- 1. RE-DECLARAÇÃO DAS ARQUITETURAS (devem ser idênticas aos scripts de treino) ---

class ConditionalSketchGenerator(nn.Module):
    def __init__(self):
        super(ConditionalSketchGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, condition):
        x = self.fc(condition)
        x = x.view(-1, 32, 4, 4)
        return self.deconv(x)


class MutationCNN(nn.Module):
    def __init__(self):
        super(MutationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
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
        return torch.clamp(x + delta * 0.5, 0.0, 1.0)


# --- 2. FUNÇÃO AUXILIAR DE VISUALIZAÇÃO ---
def print_level(level, title="Labirinto"):
    """Imprime o labirinto no terminal usando caracteres visuais."""
    print(f"\n--- {title} ---")
    for row in level:
        row_str = "".join(["██" if tile == 1 else "  " for tile in row])
        print(row_str)
    print("-" * (len(title) + 8))


def binarize_probabilistic(probs, seed=None):
    """
    Binarização probabilística: cada tile é amostrado com probabilidade = valor da rede.
    Evita o colapso para ponto fixo que ocorre com threshold hard (> 0.5).
    """
    rng = np.random.default_rng(seed)
    binary = (rng.random(probs.shape) < probs).astype(np.float32)
    # Garante que início (0,0) e fim (9,9) são sempre caminhos livres
    binary[0, 0] = 0.0
    binary[9, 9] = 0.0
    return binary


# --- 3. PIPELINE DE INFERÊNCIA COMBINADO ---
def generate_custom_level(target_density, target_complexity, mutation_steps=3):
    device = get_device()

    # 1. Carregar o Gerador de Esboços
    sketch_net = ConditionalSketchGenerator().to(device)
    try:
        sketch_net.load_state_dict(torch.load(get_sketch_model_path(), map_location=device))
        sketch_net.eval()
    except FileNotFoundError:
        print(f"Erro: '{get_sketch_model_path()}' não encontrado. Execute o Passo 3!")
        return

    # 2. Carregar o Modelo de Mutação
    mutation_net = MutationCNN().to(device)
    try:
        mutation_net.load_state_dict(torch.load(get_mutation_model_path(), map_location=device))
        mutation_net.eval()
    except FileNotFoundError:
        print(f"Erro: '{get_mutation_model_path()}' não encontrado. Execute o Passo 2!")
        return

    print(f"\n[INFO] Solicitando mapa com Densidade: {target_density*100:.0f}% e Complexidade: {target_complexity*100:.0f}%")

    # --- ETAPA A: GERANDO O ESBOÇO ---
    condition_tensor = torch.tensor([[target_density, target_complexity]], dtype=torch.float32).to(device)

    with torch.no_grad():
        sketch_output = sketch_net(condition_tensor)

    sketch_probs = sketch_output.squeeze().cpu().numpy()
    # Binarização probabilística para o esboço inicial
    sketch_binary = binarize_probabilistic(sketch_probs)
    print_level(sketch_binary, "1. ESBOÇO INICIAL (Bruto)")

    # --- ETAPA B: REFINAMENTO PELO MODELO DE MUTAÇÃO ---
    current_map_tensor = torch.tensor(sketch_binary, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    for step in range(mutation_steps):
        with torch.no_grad():
            mutation_output = mutation_net(current_map_tensor)

        mutated_probs = mutation_output.squeeze().cpu().numpy()
        # Amostragem probabilística: cada chamada pode gerar resultado diferente
        sketch_binary = binarize_probabilistic(mutated_probs, seed=None)

        current_map_tensor = torch.tensor(sketch_binary, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    print_level(sketch_binary, f"2. MAPA REFINADO (Após {mutation_steps} Mutações)")
    return sketch_binary


if __name__ == "__main__":
    DESIRED_DENSITY = 0.30
    DESIRED_COMPLEXITY = 0.50

    generate_custom_level(target_density=DESIRED_DENSITY,
                          target_complexity=DESIRED_COMPLEXITY,
                          mutation_steps=4)