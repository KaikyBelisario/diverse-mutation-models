import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# --- CONFIGURAÇÕES DO AMBIENTE ---
GRID_SIZE = 10         # Labirinto 10x10
ARCHIVE_RES = 10       # Resolução da Grelha MAP-Elites (10x10 células)
NUM_ITERATIONS = 5000  # Quantas mutações vamos testar

# --- FUNÇÕES AUXILIARES DE JOGABILIDADE (FITNESS) ---
def bfs_shortest_path(level):
    """
    Calcula o caminho mais curto do topo-esquerdo (0,0) ao canto inferior-direito (9,9).
    Retorna o comprimento do caminho. Se não houver caminho, retorna 0.
    O nível contém: 0 para caminho livre, 1 para parede.
    """
    if level[0, 0] == 1 or level[GRID_SIZE-1, GRID_SIZE-1] == 1:
        return 0
    
    start = (0, 0)
    end = (GRID_SIZE-1, GRID_SIZE-1)
    queue = deque([[start]])
    seen = {start}
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == end:
            return len(path)
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and level[nx, ny] == 0 and (nx, ny) not in seen:
                queue.append(path + [(nx, ny)])
                seen.add((nx, ny))
    return 0

def evaluate_level(level):
    """
    Avalia o nível e retorna (Fitness, BC1, BC2)
    BC1: Densidade de Paredes (0.0 a 1.0)
    BC2: Rugosidade/Mudanças de blocos horizontais (Complexidade Visual)
    """
    fitness = bfs_shortest_path(level) # Qualidade = tamanho do caminho resolvível
    
    # Característica Comportamental 1: Densidade
    bc1 = np.sum(level) / (GRID_SIZE * GRID_SIZE)
    
    # Característica Comportamental 2: Transições horizontais (0.0 a 1.0)
    bc2 = np.sum(level[:, :-1] != level[:, 1:]) / (GRID_SIZE * (GRID_SIZE - 1))
    
    return fitness, bc1, bc2

# --- OPERADORES EVOLUTIVOS ---
def generate_random_level():
    """Gera um nível puramente aleatório. VERIFICAR O RANDOM DO NUMPY"""
    level = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.7, 0.3])
    level[0, 0] = 0
    level[GRID_SIZE-1, GRID_SIZE-1] = 0
    return level

def mutate(level):
    """Aplica uma mutação ruidosa (inverte de 1 a 3 blocos aleatórios)."""
    new_level = level.copy()
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if (x, y) != (0, 0) and (x, y) != (GRID_SIZE-1, GRID_SIZE-1):
            new_level[x, y] = 1 - new_level[x, y] # Inverte 0->1 ou 1->0
    return new_level

# --- MAP-ELITES COM CAPTURA DE DATASET ---
class MAPElites:
    def __init__(self):
        # Guarda o nível de elite para cada célula da grelha
        self.archive_levels = {}
        # Guarda a fitness correspondente de cada célula
        self.archive_fitness = np.zeros((ARCHIVE_RES, ARCHIVE_RES))
        # O DATASET FUSÃO: Guardará pares de (Pai/Estado Ruidoso, Filho/Estado Melhorado)
        self.dataset_transitions = []

    def get_grid_indices(self, bc1, bc2):
        """Mapeia os valores contínuos de BC1 e BC2 para índices discretos da grelha (0 a ARCHIVE_RES-1)."""
        idx_x = int(bc1 * (ARCHIVE_RES - 1))
        idx_y = int(bc2 * (ARCHIVE_RES - 1))
        return clamp(idx_x, 0, ARCHIVE_RES-1), clamp(idx_y, 0, ARCHIVE_RES-1)

    def run(self):
        print("-> Inicializando Arquivo MAP-Elites com níveis aleatórios...")
        for _ in range(100):
            lvl = generate_random_level()
            fit, bc1, bc2 = evaluate_level(lvl)
            if fit > 0: # Apenas aceita níveis jogáveis inicialmente
                ix, iy = self.get_grid_indices(bc1, bc2)
                if fit > self.archive_fitness[ix, iy]:
                    self.archive_fitness[ix, iy] = fit
                    self.archive_levels[(ix, iy)] = lvl

        print(f"-> Executando {NUM_ITERATIONS} iterações de busca por Diversidade e Qualidade...")
        for i in range(NUM_ITERATIONS):
            if not self.archive_levels:
                continue
                
            # 1. Seleciona um pai aleatório dos Elites já existentes (Conceito MLQD)
            parent_key = random.choice(list(self.archive_levels.keys()))
            parent_lvl = self.archive_levels[parent_key]
            parent_fit = self.archive_fitness[parent_key]
            
            # 2. Mutaciona o pai
            child_lvl = mutate(parent_lvl)
            child_fit, child_bc1, child_bc2 = evaluate_level(child_lvl)
            
            if child_fit > 0: # Critério de viabilidade
                ix, iy = self.get_grid_indices(child_bc1, child_bc2)
                
                # Se a célula estiver vazia OU o filho for melhor que o elite atual daquela célula
                if child_fit > self.archive_fitness[ix, iy]:
                    
                    # SALVA A TRANSIÇÃO (Conceito Mutation Models!)
                    # O modelo de mutação precisa aprender como transformar o pai no filho bem-sucedido
                    self.dataset_transitions.append({
                        'parent': parent_lvl.tolist(),
                        'child': child_lvl.tolist()
                    })
                    
                    # Atualiza a grelha de elites
                    self.archive_fitness[ix, iy] = child_fit
                    self.archive_levels[(ix, iy)] = child_lvl

        print(f"-> Concluído! Transições evolutivas capturadas: {len(self.dataset_transitions)}")
        return self.dataset_transitions

def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)

# --- EXECUÇÃO E SALVAMENTO ---
if __name__ == "__main__":
    optimizer = MAPElites()
    dataset = optimizer.run()
    
    # Salva o dataset gerado em formato binário do NumPy para o Passo 2
    np.save("dataset_mutacoes_qd.npy", dataset)
    print("-> Dataset salvo com sucesso como 'dataset_mutacoes_qd.npy'!")
    
    # Visualização da Grelha de Qualidade e Diversidade obtida
    plt.figure(figsize=(6,5))
    plt.imshow(optimizer.archive_fitness, origin='lower', cmap='viridis')
    plt.colorbar(label='Fitness (Comprimento do Caminho Solúvel)')
    plt.title('Espaço de Qualidade-Diversidade (MAP-Elites)')
    plt.xlabel('Densidade de Paredes (BC1)')
    plt.ylabel('Complexidade Visual (BC2)')
    plt.show()