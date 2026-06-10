"""
domain.py — Domínio: Mapas Estratégicos 8×8
--------------------------------------------
Define representação, restrições, inicialização, mutação e as 16 métricas
de comportamento usadas pelo CPA/MLQD (Sfikas et al., 2025 §4.1).

Tiles: Floor=0, Wall=1, Resource=2, Base=3
"""

import numpy as np
from collections import deque

# ── Tipos de tile ──────────────────────────────────────────────────────────────
FLOOR    = 0
WALL     = 1
RESOURCE = 2
BASE     = 3

GRID_SIZE   = 8
N_TILES     = GRID_SIZE * GRID_SIZE   # 64
N_FEATURES  = 16
CROP_SIZE   = 8   # Janela local para o modelo de mutação


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de BFS / pathfinding
# ─────────────────────────────────────────────────────────────────────────────

def _neighbors(r, c):
    """Vizinhos ortogonais válidos de (r, c)."""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            yield nr, nc


def bfs_distances(level, start_r, start_c):
    """
    BFS de (start_r, start_c) passando apenas por tiles não-Wall.
    Retorna grid [GRID_SIZE, GRID_SIZE] com distâncias (-1 = inacessível).
    """
    dist = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int32)
    if level[start_r, start_c] == WALL:
        return dist
    dist[start_r, start_c] = 0
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c):
            if dist[nr, nc] == -1 and level[nr, nc] != WALL:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _passable_tiles(level):
    return [(r, c) for r in range(GRID_SIZE)
            for c in range(GRID_SIZE) if level[r, c] != WALL]


def _tiles_of_type(level, tile_type):
    return [(r, c) for r in range(GRID_SIZE)
            for c in range(GRID_SIZE) if level[r, c] == tile_type]


# ─────────────────────────────────────────────────────────────────────────────
# Restrições e Score de Viabilidade  (MLQD §4.1.2, Eq. 1)
# ─────────────────────────────────────────────────────────────────────────────

def feasibility_score(level):
    """
    Retorna (is_feasible: bool, f_inf: float ∈ [0, 1]).
    f_inf cresce à medida que o mapa se aproxima de ser viável.
    Um mapa viável tem f_inf = 1.0.

    Restrições verificadas:
      (a) Exatamente 2 bases
      (b) Entre 4 e 10 recursos
      (c) Todos os tiles especiais conectados (verificado via BFS)
    """
    bases     = _tiles_of_type(level, BASE)
    resources = _tiles_of_type(level, RESOURCE)

    b = len(bases)
    r = len(resources)

    if b != 2 or not (4 <= r <= 10):
        return False, 0.0

    # cb: pares ORDENADOS de bases conectados (para b=2, máximo = b*(b-1) = 2).
    # Como o grafo é não-direcionado, (b0→b1) implica (b1→b0), então cb = 0 ou 2.
    d = bfs_distances(level, bases[0][0], bases[0][1])
    cb = 2 if d[bases[1][0], bases[1][1]] >= 0 else 0

    # cr: pares (base, recurso) conectados
    cr = 0
    for br, bc_c in bases:
        db = bfs_distances(level, br, bc_c)
        for rr, rc in resources:
            if db[rr, rc] >= 0:
                cr += 1

    b_pairs  = b * (b - 1)   # = 2
    br_pairs = r * b          # = r * 2

    f_inf = 0.5 * (cb / b_pairs) + 0.5 * (cr / br_pairs)
    return (f_inf >= 1.0), float(f_inf)


# ─────────────────────────────────────────────────────────────────────────────
# Inicialização  (MLQD §4.1.3)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_map():
    """
    Gera mapa viável inicial: tudo Floor, 2 Bases aleatórias, 4-10 Resources.
    Sem muros → conectividade garantida por construção.
    """
    level = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # 2 Bases em posições distintas
    base_indices = np.random.choice(N_TILES, size=2, replace=False)
    for idx in base_indices:
        r, c = divmod(int(idx), GRID_SIZE)
        level[r, c] = BASE

    # 4–10 Resources em tiles Floor restantes
    n_res = np.random.randint(4, 11)
    floor_positions = [(r, c) for r in range(GRID_SIZE)
                       for c in range(GRID_SIZE) if level[r, c] == FLOOR]
    chosen = np.random.choice(len(floor_positions), size=n_res, replace=False)
    for idx in chosen:
        r, c = floor_positions[idx]
        level[r, c] = RESOURCE

    return level


# ─────────────────────────────────────────────────────────────────────────────
# Repair de Constraints pós-geração  (chamado após Transformer.generate)
# ─────────────────────────────────────────────────────────────────────────────

def repair_constraints(level: np.ndarray) -> np.ndarray:
    """
    Garante constraints (a) e (b) após geração pelo Transformer:
      (a) Exatamente 2 Bases
      (b) Entre 4 e 10 Resources

    O Transformer não aprende que resources são obrigatórios (tokens raros,
    ~11% da sequência) — este repair injeta os tiles necessários antes que
    o MutationCNN tente corrigir a conectividade (constraint c).
    """
    level = level.copy()

    def _pos(tile):
        return [(r, c) for r in range(GRID_SIZE)
                for c in range(GRID_SIZE) if level[r, c] == tile]

    # ── (a) Exatamente 2 Bases ────────────────────────────────────────────────
    bases = _pos(BASE)
    if len(bases) > 2:
        for r, c in bases[2:]:
            level[r, c] = FLOOR
    elif len(bases) < 2:
        needed = 2 - len(bases)
        for src in (FLOOR, WALL):
            pool = list(_pos(src))
            np.random.shuffle(pool)
            for r, c in pool[:needed]:
                level[r, c] = BASE
                needed -= 1
            if needed <= 0:
                break

    # ── (b) 4–10 Resources ────────────────────────────────────────────────────
    resources = _pos(RESOURCE)
    if len(resources) > 10:
        for r, c in resources[10:]:
            level[r, c] = FLOOR
    elif len(resources) < 4:
        needed = 4 - len(resources)
        for src in (FLOOR, WALL):
            pool = [p for p in _pos(src) if level[p[0], p[1]] != BASE]
            np.random.shuffle(pool)
            for r, c in pool[:needed]:
                level[r, c] = RESOURCE
                needed -= 1
            if needed <= 0:
                break

    return level


# ─────────────────────────────────────────────────────────────────────────────
# Mutação  (MLQD §4.1.4)
# ─────────────────────────────────────────────────────────────────────────────

def mutate_map(level, mutation_model=None):
    """
    Mutação conforme MLQD §4.1.4:
      - Seleciona 5–20 % dos tiles (excluindo Bases).
      - Para cada tile: swap com vizinho adjacente  OU  toggle Wall↔Floor.
      - Resources só são removidos se a contagem ficaria acima de 4.

    Se mutation_model não é None (Assisted Evolution), usa o modelo para
    decidir a ação com 75 % de probabilidade (25 % aleatório — exploração).

    Retorna:
      new_level : np.int8 array 8×8
      changes   : lista de (r, c, new_tile_value) para cada tile efetivamente alterado
    """
    new_level = level.copy()
    n_mut = max(1, int(np.random.uniform(0.05, 0.20) * N_TILES))

    candidates = [(r, c) for r in range(GRID_SIZE)
                  for c in range(GRID_SIZE) if new_level[r, c] != BASE]
    np.random.shuffle(candidates)
    candidates = candidates[:n_mut]

    changes = []

    for r, c in candidates:
        old_tile = new_level[r, c]

        if mutation_model is not None and np.random.random() < 0.75:
            action = _apply_model_mutation(mutation_model, new_level, r, c)
        else:
            action = np.random.choice(['swap', 'toggle'])

        if action == 'swap':
            nbrs = list(_neighbors(r, c))
            if not nbrs:
                continue
            nr, nc = nbrs[np.random.randint(len(nbrs))]
            if new_level[nr, nc] == BASE:
                continue
            new_level[r, c], new_level[nr, nc] = new_level[nr, nc], new_level[r, c]
            if new_level[r, c] != old_tile:
                changes.append((r, c, int(new_level[r, c])))
            if new_level[nr, nc] != level[nr, nc]:
                changes.append((nr, nc, int(new_level[nr, nc])))

        elif action == 'toggle':
            if new_level[r, c] == WALL:
                new_level[r, c] = FLOOR
            elif new_level[r, c] == FLOOR:
                new_level[r, c] = WALL
            if new_level[r, c] != old_tile:
                changes.append((r, c, int(new_level[r, c])))

        elif isinstance(action, (int, np.integer)):
            # Saída do modelo: tile alvo
            if action == RESOURCE and np.sum(new_level == RESOURCE) >= 10:
                continue
            if new_level[r, c] == RESOURCE and action != RESOURCE:
                if np.sum(new_level == RESOURCE) <= 4:
                    continue
            new_level[r, c] = int(action)
            if new_level[r, c] != old_tile:
                changes.append((r, c, int(new_level[r, c])))

    return new_level, changes


def _apply_model_mutation(model, level, r, c):
    """Usa o MutationCNN para prever ação em (r, c). Retorna tile ou 'no_change'."""
    import torch
    crop = crop_around(level, r, c)
    device = next(model.parameters()).device
    inp = torch.tensor(crop / 3.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        action_idx = logits.argmax(dim=1).item()
    # 0=NoChange, 1=Floor, 2=Wall, 3=Resource
    action_map = {0: 'no_change', 1: FLOOR, 2: WALL, 3: RESOURCE}
    return action_map[action_idx]


def crop_around(level, r, c, size=CROP_SIZE):
    """
    Recorta janela size×size centrada em (r, c) com padding WALL fora dos limites.
    Usado como input do MutationCNN (narrow representation — Khalifa et al. §4.3).
    """
    half = size // 2
    padded = np.pad(level, half, mode='constant', constant_values=WALL)
    pr, pc = r + half, c + half
    return padded[pr - half: pr + half, pc - half: pc + half].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# As 16 Métricas de Comportamento  (MLQD §4.1.5, Tabela 1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(level):
    """
    Retorna array de 16 features ∈ [0, 1]:

    F1  – Floor Ratio
    F2  – Wall Ratio
    F3  – Resource Ratio
    F4  – Horizontal Symmetry
    F5  – Vertical Symmetry
    F6  – Diagonal Symmetry
    F7  – Anti-diagonal Symmetry
    F8  – Wall Islands Ratio
    F9  – Passable Graph Diameter Ratio
    F10 – Bases Distance Ratio
    F11 – Resource Safety
    F12 – Resource Safety Balance
    F13 – Base Safety
    F14 – Base Safety Balance
    F15 – Exploration
    F16 – Exploration Balance
    """
    # F1–F3: ratios de tile
    F1 = np.sum(level == FLOOR)    / N_TILES
    F2 = np.sum(level == WALL)     / N_TILES
    F3 = np.sum(level == RESOURCE) / N_TILES

    # F4–F7: simetrias
    F4 = _symmetry(level, 'horizontal')
    F5 = _symmetry(level, 'vertical')
    F6 = _symmetry(level, 'diagonal')
    F7 = _symmetry(level, 'antidiagonal')

    # F8: wall islands
    F8 = _wall_islands_ratio(level)

    # F9: diâmetro do grafo passável
    passable = _passable_tiles(level)
    F9 = _passable_graph_diameter(level, passable) / (N_TILES - 1) if len(passable) >= 2 else 0.0

    # F10: distância BFS entre as 2 bases
    bases = _tiles_of_type(level, BASE)
    if len(bases) == 2:
        d = bfs_distances(level, bases[0][0], bases[0][1])
        raw = d[bases[1][0], bases[1][1]]
        F10 = (raw / (N_TILES - 1)) if raw >= 0 else 0.0
    else:
        F10 = 0.0

    # F11–F16: métricas estratégicas
    F11, F12, F13, F14, F15, F16 = _strategy_metrics(level, bases, passable)

    feats = np.array([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                      F11, F12, F13, F14, F15, F16], dtype=np.float32)
    return np.clip(feats, 0.0, 1.0)


# ── Funções auxiliares das features ──────────────────────────────────────────

def _symmetry(level, kind):
    """Fração de tiles que coincidem com seu par simétrico (MLQD Tabela 1)."""
    match = 0
    g = GRID_SIZE
    for r in range(g):
        for c in range(g):
            if kind == 'horizontal':
                nr, nc = r, g - 1 - c
            elif kind == 'vertical':
                nr, nc = g - 1 - r, c
            elif kind == 'diagonal':
                nr, nc = c, r
            else:  # antidiagonal
                nr, nc = g - 1 - c, g - 1 - r
            if level[r, c] == level[nr, nc]:
                match += 1
    return match / N_TILES


def _wall_islands_ratio(level):
    """
    F8 = 2 · I_w / N  (MLQD Tabela 1).
    I_w = número de componentes conectados de Wall tiles.
    """
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    islands = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if level[r, c] == WALL and not visited[r, c]:
                islands += 1
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in _neighbors(cr, cc):
                        if level[nr, nc] == WALL and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
    return min(2 * islands / N_TILES, 1.0)


def _passable_graph_diameter(level, passable):
    """
    Diâmetro do grafo passável via duplo-BFS (F9).
    O duplo-BFS devolve o diâmetro exato para grafos não-pesados.
    """
    if len(passable) < 2:
        return 0
    # 1ª BFS de qualquer ponto
    r0, c0 = passable[0]
    d = bfs_distances(level, r0, c0)
    reachable = [(r, c) for r, c in passable if d[r, c] >= 0]
    if not reachable:
        return 0
    # Ponto mais distante
    r1, c1 = max(reachable, key=lambda p: d[p[0]][p[1]])
    # 2ª BFS desse ponto
    d2 = bfs_distances(level, r1, c1)
    return max((d2[r, c] for r, c in reachable if d2[r, c] >= 0), default=0)


def _strategy_metrics(level, bases, passable):
    """
    F11–F16: métricas para qualidade estratégica do mapa.
    Inspiradas em Sfikas et al. [19] (recurso de segurança, exploração etc.).
    """
    resources = _tiles_of_type(level, RESOURCE)

    if len(bases) != 2 or not resources or not passable:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    d_b0 = bfs_distances(level, bases[0][0], bases[0][1])
    d_b1 = bfs_distances(level, bases[1][0], bases[1][1])

    def dist(d, r, c):
        v = d[r, c]
        return v if v >= 0 else N_TILES

    # F11: resource safety — distância média mínima de cada recurso à base mais próxima
    min_dists = [min(dist(d_b0, rr, rc), dist(d_b1, rr, rc)) for rr, rc in resources]
    F11 = float(np.mean(min_dists) / (N_TILES - 1))

    # F12: resource safety balance — equilíbrio de recursos entre as duas bases
    closer_b0 = sum(1 for rr, rc in resources
                    if dist(d_b0, rr, rc) <= dist(d_b1, rr, rc))
    n_r = len(resources)
    F12 = 1.0 - abs(closer_b0 - (n_r - closer_b0)) / n_r

    # F13: base safety — fração de tiles passáveis "dominados" perto de alguma base
    safe_thresh = (N_TILES - 1) * 0.25
    n_safe = sum(1 for r, c in passable
                 if min(dist(d_b0, r, c), dist(d_b1, r, c)) <= safe_thresh)
    F13 = n_safe / len(passable)

    # F14: base safety balance — equilíbrio do território dominado por cada base
    dom_b0 = sum(1 for r, c in passable if dist(d_b0, r, c) < dist(d_b1, r, c))
    n_p = len(passable)
    F14 = 1.0 - abs(dom_b0 - (n_p - dom_b0)) / n_p

    # F15: exploration — alcance máximo a partir de qualquer base (normalizado)
    max_b0 = max((d_b0[r, c] for r, c in passable if d_b0[r, c] >= 0), default=0)
    max_b1 = max((d_b1[r, c] for r, c in passable if d_b1[r, c] >= 0), default=0)
    F15 = max(max_b0, max_b1) / (N_TILES - 1)

    # F16: exploration balance — equilíbrio do alcance entre as duas bases
    F16 = 1.0 - abs(max_b0 - max_b1) / (N_TILES - 1)

    return float(F11), float(F12), float(F13), float(F14), float(F15), float(F16)

