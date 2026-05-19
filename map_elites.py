import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from path_utils import ensure_results_dir, get_dataset_path, get_qd_map_path

# --- CONFIGURAÇÕES DO AMBIENTE ---
GRID_SIZE = 10         # Labirinto 10x10
ARCHIVE_RES = 10       # Resolução da Grelha MAP-Elites (10x10 células)
NUM_ITERATIONS = 20000  # Quantas mutações vamos testar

# --- FUNÇÕES AUXILIARES DE JOGABILIDADE (FITNESS) ---
def bfs_shortest_path(level):
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
    fitness = bfs_shortest_path(level) 
    bc1 = np.sum(level) / (GRID_SIZE * GRID_SIZE)
    bc2 = np.sum(level[:, :-1] != level[:, 1:]) / (GRID_SIZE * (GRID_SIZE - 1))
    return fitness, bc1, bc2

# --- OPERADORES EVOLUTIVOS ---
def generate_random_level():
    level = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.7, 0.3])
    level[0, 0] = 0
    level[GRID_SIZE-1, GRID_SIZE-1] = 0
    return level

def mutate(level):
    new_level = level.copy()
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if (x, y) != (0, 0) and (x, y) != (GRID_SIZE-1, GRID_SIZE-1):
            new_level[x, y] = 1 - new_level[x, y]
    return new_level

# --- MAP-ELITES COM CAPTURA DE DATASET ---
class MAPElites:
    def __init__(self):
        self.archive_levels = {}
        self.archive_fitness = np.zeros((ARCHIVE_RES, ARCHIVE_RES))
        self.dataset_transitions = []

    def get_grid_indices(self, bc1, bc2):
        idx_x = int(bc1 * (ARCHIVE_RES - 1))
        idx_y = int(bc2 * (ARCHIVE_RES - 1))
        return clamp(idx_x, 0, ARCHIVE_RES-1), clamp(idx_y, 0, ARCHIVE_RES-1)

    def run(self):
        print("-> Inicializando Arquivo MAP-Elites...")
        for _ in range(100):
            lvl = generate_random_level()
            fit, bc1, bc2 = evaluate_level(lvl)
            if fit > 0: 
                ix, iy = self.get_grid_indices(bc1, bc2)
                if fit > self.archive_fitness[ix, iy]:
                    self.archive_fitness[ix, iy] = fit
                    self.archive_levels[(ix, iy)] = lvl

        print(f"-> Executando {NUM_ITERATIONS} iterações de busca...")
        for i in range(NUM_ITERATIONS):
            if not self.archive_levels:
                continue
                
            parent_key = random.choice(list(self.archive_levels.keys()))
            parent_lvl = self.archive_levels[parent_key]
            
            child_lvl = mutate(parent_lvl)
            child_fit, child_bc1, child_bc2 = evaluate_level(child_lvl)
            
            # CORREÇÃO AQUI: Se o filho for um mapa jogável, ele entra para o Dataset!
            if child_fit > 0: 
                ix, iy = self.get_grid_indices(child_bc1, child_bc2)
                
                # Captura a transição para dar volume massivo de treino à IA
                self.dataset_transitions.append({
                    'parent': parent_lvl.tolist(),
                    'child': child_lvl.tolist()
                })
                
                # A grelha do MAP-Elites continua guardando rigidamente apenas os melhores
                if child_fit > self.archive_fitness[ix, iy]:
                    self.archive_fitness[ix, iy] = child_fit
                    self.archive_levels[(ix, iy)] = child_lvl

        print(f"-> Concluído! Transições evolutivas capturadas: {len(self.dataset_transitions)}")
        return self.dataset_transitions

def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)

if __name__ == "__main__":
    ensure_results_dir()

    optimizer = MAPElites()
    dataset = optimizer.run()

    np.save(get_dataset_path(), dataset)
    print(f"-> Dataset salvo com sucesso em '{get_dataset_path()}'!")

    plt.figure(figsize=(6, 5))
    plt.imshow(optimizer.archive_fitness, origin='lower', cmap='viridis')
    plt.colorbar(label='Fitness')
    plt.title('Espaço de Qualidade-Diversidade')
    plt.savefig(get_qd_map_path(), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico salvo em '{get_qd_map_path()}'")
