"""
cpa.py — Cross-Pollination of Axis-Aligned Archives (CPA / FI-CPA)
--------------------------------------------------------------------
Implementa o FI-CPA conforme MLQD §3.1 e §5.1.1 (Sfikas et al., 2025).

Estrutura:
  - 16 arquivos 1D feasíveis  (um por feature F1–F16, N_BINS células cada)
  - 1  arquivo 1D infeasível  (por f_inf)
  - EHR (Evolutionary History Record): todas as triplas
    {parent, loc_r, loc_c, action} de indivíduos viáveis

Suporte a Assisted Evolution (Khalifa et al., 2022 §3.2):
  - Quando retrain_interval e retrain_callback forem fornecidos,
    o modelo de mutação é re-treinado a cada N passos e passa a ser
    usado como operador de mutação.
"""

import numpy as np
import random
from domain import (
    N_FEATURES, N_TILES,
    initialize_map, mutate_map, feasibility_score, compute_features,
)
from path_utils import ensure_results_dir, get_dataset_path

# ── Hiperparâmetros do CPA ─────────────────────────────────────────────────────
N_BINS              = 65       # Bins por arquivo 1D  (MLQD §5.1.1)
N_FEASIBLE_ARCHIVES = N_FEATURES   # 16 arquivos
INIT_POP_SIZE       = 1_105    # ≈ N_bins * (N_arq + 1) — MLQD §5.1.1

# Ajuste conforme capacidade computacional:
#   Paper usa 10^6; para testes rápidos use 10_000–50_000.
DEFAULT_TARGET_EHR  = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Arquivo 1D
# ─────────────────────────────────────────────────────────────────────────────

class Archive1D:
    """
    Arquivo 1D com N_BINS células.
    Cada célula guarda o mapa cujo valor de feature cai naquele bin.
    Bin de índice k cobre o intervalo [k/(N-1), (k+1)/(N-1)).
    Substituição: ocorre sempre (exploração pura, sem critério de qualidade extra).
    """
    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.cells  = {}   # bin_idx -> np.array (mapa int8)

    def _bin(self, value: float) -> int:
        return int(np.clip(round(value * (self.n_bins - 1)), 0, self.n_bins - 1))

    def update(self, level: np.ndarray, feature_value: float) -> None:
        idx = self._bin(feature_value)
        self.cells[idx] = level.copy()

    def sample(self) -> np.ndarray | None:
        if not self.cells:
            return None
        return self.cells[random.choice(list(self.cells.keys()))].copy()

    def __len__(self) -> int:
        return len(self.cells)


# ─────────────────────────────────────────────────────────────────────────────
# Algoritmo FI-CPA
# ─────────────────────────────────────────────────────────────────────────────

class FICPA:
    """
    Feasible-Infeasible CPA.

    Parâmetros
    ----------
    target_ehr        : int   — tamanho alvo do EHR (parar quando atingido)
    mutation_model    : nn.Module | None — modelo inicial para Assisted Evolution
    retrain_interval  : int | None — re-treina a cada N passos (Assisted)
    retrain_callback  : callable | None — fn(ehr) -> nn.Module (re-treino externo)
    """

    def __init__(self,
                 target_ehr: int = DEFAULT_TARGET_EHR,
                 mutation_model=None,
                 retrain_interval: int | None = None,
                 retrain_callback=None):
        self.feasible_archives  = [Archive1D() for _ in range(N_FEASIBLE_ARCHIVES)]
        self.infeasible_archive = Archive1D()
        self.ehr                = []
        self.target_ehr         = target_ehr
        self.mutation_model     = mutation_model
        self.retrain_interval   = retrain_interval
        self.retrain_callback   = retrain_callback

    # ── Colocação nos arquivos ─────────────────────────────────────────────────

    def _place_feasible(self, level: np.ndarray, features: np.ndarray) -> None:
        """
        Seleciona um arquivo aleatório e coloca o indivíduo nele
        (cross-pollination — MLQD §3.1, passo 3).
        """
        arch_idx = random.randrange(N_FEASIBLE_ARCHIVES)
        self.feasible_archives[arch_idx].update(level, float(features[arch_idx]))

    def _place_infeasible(self, level: np.ndarray, f_inf: float) -> None:
        self.infeasible_archive.update(level, f_inf)

    # ── Registro no EHR ───────────────────────────────────────────────────────

    def _record_ehr(self, parent: np.ndarray, child: np.ndarray, changes: list) -> None:
        """
        Adiciona uma entrada ao EHR por cada tile alterado durante a mutação.
        Cada entrada inclui o mapa filho completo (para o Transformer) e o
        contexto local de cada mudança (para o MutationCNN).
        """
        child_list = child.tolist()   # serializado uma vez, reutilizado por cada change
        parent_list = parent.tolist()
        for r, c, new_tile in changes:
            self.ehr.append({
                'parent': parent_list,
                'child' : child_list,   # mapa filho completo — usado pelo Transformer
                'loc_r' : r,
                'loc_c' : c,
                'action': new_tile,     # tile resultante — usado pelo MutationCNN
            })

    # ── Seleção de pai ────────────────────────────────────────────────────────

    def _select_parent(self) -> np.ndarray:
        """
        Alterna entre arquivos feasível e infeasível para selecionar o pai
        (FI-CPA §5.1.1 — 50% de cada).
        """
        use_feasible = random.random() < 0.5
        if use_feasible:
            non_empty = [a for a in self.feasible_archives if len(a) > 0]
            if non_empty:
                return random.choice(non_empty).sample()
        if len(self.infeasible_archive) > 0:
            return self.infeasible_archive.sample()
        # Fallback: inicialização aleatória
        return initialize_map()

    # ── Loop principal ────────────────────────────────────────────────────────

    def run(self) -> list:
        """
        Executa o FI-CPA até coletar target_ehr indivíduos viáveis no EHR.
        Retorna o EHR como lista de dicts.
        """
        # ── Fase de Inicialização ─────────────────────────────────────────────
        print(f"[CPA] Inicializando com {INIT_POP_SIZE} indivíduos...")
        for _ in range(INIT_POP_SIZE):
            lvl = initialize_map()
            is_feas, f_inf = feasibility_score(lvl)
            if is_feas:
                feats = compute_features(lvl)
                self._place_feasible(lvl, feats)
            else:
                self._place_infeasible(lvl, f_inf)

        feasible_cells = sum(len(a) for a in self.feasible_archives)
        print(f"[CPA] Arquivo viável: {feasible_cells} células | "
              f"Infeasível: {len(self.infeasible_archive)} células")
        print(f"[CPA] Coletando {self.target_ehr} entradas no EHR...")

        # ── Fase de Operação Core ─────────────────────────────────────────────
        step = 0
        while len(self.ehr) < self.target_ehr:
            step += 1
            parent = self._select_parent()
            child, changes = mutate_map(parent, self.mutation_model)

            is_feas, f_inf = feasibility_score(child)

            if is_feas:
                feats = compute_features(child)
                self._place_feasible(child, feats)
                if changes:
                    self._record_ehr(parent, child, changes)
            else:
                self._place_infeasible(child, f_inf)

            # ── Assisted Evolution: re-treino periódico ───────────────────────
            if (self.retrain_interval and self.retrain_callback
                    and step % self.retrain_interval == 0
                    and len(self.ehr) >= 500):
                print(f"[CPA] Assisted re-treino (step={step:,}, ehr={len(self.ehr):,})...")
                self.mutation_model = self.retrain_callback(self.ehr)

            # ── Log de progresso ──────────────────────────────────────────────
            if step % 5_000 == 0:
                feas_cells = sum(len(a) for a in self.feasible_archives)
                print(f"   step={step:>8,} | EHR={len(self.ehr):>6,} / {self.target_ehr:,} "
                      f"| cells_feas={feas_cells}")

        print(f"[CPA] Concluído! EHR com {len(self.ehr):,} entradas "
              f"({step:,} iterações no total).")
        return self.ehr

    # ── Propriedades úteis ────────────────────────────────────────────────────

    @property
    def feasible_cell_count(self) -> int:
        return sum(len(a) for a in self.feasible_archives)

    def get_feasible_samples(self, n: int = 100) -> list:
        """Retorna até n mapas viáveis amostrados dos arquivos."""
        samples = []
        for _ in range(n):
            non_empty = [a for a in self.feasible_archives if len(a) > 0]
            if non_empty:
                lvl = random.choice(non_empty).sample()
                if lvl is not None:
                    samples.append(lvl)
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_results_dir()
    cpa = FICPA(target_ehr=DEFAULT_TARGET_EHR)
    ehr = cpa.run()
    np.save(get_dataset_path(), ehr)
    print(f"[CPA] Dataset salvo em '{get_dataset_path()}' "
          f"({len(ehr):,} entradas).")



