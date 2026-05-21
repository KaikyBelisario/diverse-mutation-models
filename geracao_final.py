"""
geracao_final.py — Pipeline de Inferência: Transformer + MutationCNN
---------------------------------------------------------------------
Passo 4 do pipeline PCGML:
  1. MapTransformer gera um mapa 8×8 autoregressivamente (top-p sampling)
  2. MutationCNN refina o mapa via scanline (narrow representation)

Nenhuma função de fitness é usada durante a inferência
(Khalifa et al. §4.4 — modelo inferência independente de fitness).
"""

import torch
import numpy as np

from device_utils import get_device
from path_utils import get_transformer_model_path, get_mutation_model_path
from domain import (
    feasibility_score, compute_features,
    GRID_SIZE, FLOOR, WALL, RESOURCE, BASE, crop_around, CROP_SIZE,
    repair_constraints,
)
from transformer_model import MapTransformer
from treino_mutacao import MutationCNN, N_ACTIONS, _TILE_TO_ACTION

# Mapeamento ação → tile  (inverso de _TILE_TO_ACTION)
_ACTION_TO_TILE = {v: k for k, v in _TILE_TO_ACTION.items()}
_ACTION_TO_TILE[0] = None   # NoChange

TILE_SYMBOLS = {FLOOR: '  ', WALL: '██', RESOURCE: ' R', BASE: ' B'}


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de visualização
# ─────────────────────────────────────────────────────────────────────────────

def print_level(level, title: str = "Mapa") -> None:
    is_feas, f_inf = feasibility_score(level)
    print(f"\n┌─ {title} {'─' * max(0, 40 - len(title))}┐")
    for row in level:
        print('│' + ''.join(TILE_SYMBOLS.get(int(t), '??') for t in row) + '│')
    status = '✓ Viável' if is_feas else f'✗ Inviável (f_inf={f_inf:.3f})'
    print(f"└{'─' * 18}┘  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de modelos
# ─────────────────────────────────────────────────────────────────────────────

def load_transformer(device):
    model = MapTransformer().to(device)
    model.load_state_dict(torch.load(get_transformer_model_path(),
                                     map_location=device, weights_only=True))
    model.eval()
    return model


def load_mutation_cnn(device):
    model = MutationCNN().to(device)
    model.load_state_dict(torch.load(get_mutation_model_path(),
                                     map_location=device, weights_only=True))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Refinamento via MutationCNN  (scanline — Khalifa et al. §4.4)
# ─────────────────────────────────────────────────────────────────────────────

def refine_with_mutation_model(
        level:          np.ndarray,
        mutation_model: MutationCNN,
        device:         torch.device,
        steps:          int = 3,
) -> np.ndarray:
    """
    Itera sobre o mapa em scanline (row-major) por 'steps' passes completos.
    Para cada tile, pergunta ao modelo qual ação tomar;
    Bases são sempre preservadas.
    Para quando viável ou ao esgotar os steps.
    """
    current = level.copy().astype(np.float32)

    for step in range(steps):
        new_level = current.copy()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                # Bases e Resources são intocáveis:
                # - Bases são hard constraints do domínio
                # - Resources: o MutationCNN foi treinado só com changes → nunca prediz
                #   NoChange=0, o que desfaria o repair. A CNN foca em Wall/Floor (conectividade).
                if int(current[r, c]) in (BASE, RESOURCE):
                    continue

                crop = crop_around(current, r, c, size=CROP_SIZE)
                inp  = torch.tensor(
                    crop / 3.0, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_idx = mutation_model(inp).argmax(dim=1).item()

                new_tile = _ACTION_TO_TILE.get(action_idx)
                if new_tile is not None:
                    new_level[r, c] = new_tile

        current = new_level
        # Para cedo se o mapa já for viável
        is_feas, _ = feasibility_score(current.astype(np.int8))
        if is_feas:
            break

    return current.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline completo de geração
# ─────────────────────────────────────────────────────────────────────────────

def generate_map(
        mutation_steps: int   = 3,
        top_p:          float = 0.9,
        verbose:        bool  = True,
        _transformer=None,
        _mutation_cnn=None,
) -> np.ndarray:
    """
    Gera um mapa estratégico 8×8 completo:
      1. Transformer produz esboço via amostragem autoregressiva
      2. MutationCNN refina o esboço em N passes de scanline

    Aceita modelos pré-carregados opcionalmente (evita recarregar em loop).
    """
    device = get_device()

    transformer  = _transformer  or load_transformer(device)
    mutation_cnn = _mutation_cnn or load_mutation_cnn(device)

    # Etapa A: geração pelo Transformer
    sketch = transformer.generate(device, top_p=top_p)
    if verbose:
        print_level(sketch, "1. ESBOÇO (Transformer)")

    # Repair antes do refinamento: insere resources e bases em posições aleatórias
    # para que o MutationCNN trabalhe a partir de um mapa estruturalmente válido.
    sketch = repair_constraints(sketch)

    # Etapa B: refinamento pelo MutationCNN (só toca Floor/Wall — veja refine_*)
    refined = refine_with_mutation_model(sketch, mutation_cnn, device, steps=mutation_steps)

    # Repair final: re-assegura constraints após o scanline do MutationCNN
    refined = repair_constraints(refined)

    if verbose:
        print_level(refined, f"2. MAPA REFINADO ({mutation_steps} passes)")

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = get_device()

    try:
        transformer  = load_transformer(device)
        mutation_cnn = load_mutation_cnn(device)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Execute o pipeline completo primeiro (python main.py).")
        raise SystemExit(1)

    print(f"\n[Inferência] Gerando 5 mapas de exemplo...\n")
    for i in range(5):
        print(f"\n{'═'*44}")
        print(f"  Mapa {i+1}/5")
        generate_map(mutation_steps=3,
                     _transformer=transformer,
                     _mutation_cnn=mutation_cnn)
