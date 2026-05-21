"""
avaliacao.py — Métricas de Avaliação dos Modelos Gerados
---------------------------------------------------------
Implementa as hipóteses H1–H5 do MLQD (Sfikas et al., 2025 §5.2):

  H1 – Feasibility Ratio
  H2 – Unique Ratio + Unseen Ratio
  H3 – Cobertura do espaço de BCs (% de bins ocupados por feature)
  H4 – (visualizado via Expressive Range Analysis scatter 2D)
  H5 – (latência de geração — medida externamente)
"""

import numpy as np
import matplotlib.pyplot as plt
from domain import compute_features, feasibility_score, N_FEATURES


FEATURE_NAMES = [
    'Floor Ratio',           # F1
    'Wall Ratio',            # F2
    'Resource Ratio',        # F3
    'Horiz. Symmetry',       # F4
    'Vert. Symmetry',        # F5
    'Diag. Symmetry',        # F6
    'Anti-Diag. Symmetry',   # F7
    'Wall Islands',          # F8
    'Graph Diameter',        # F9
    'Bases Distance',        # F10
    'Resource Safety',       # F11
    'Resource Safety Bal.',  # F12
    'Base Safety',           # F13
    'Base Safety Bal.',      # F14
    'Exploration',           # F15
    'Exploration Bal.',      # F16
]


# ─────────────────────────────────────────────────────────────────────────────
# Função principal de avaliação
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_generated_maps(
        generated_maps: list,
        training_maps:  list | None = None,
        n_bins:         int         = 65,
) -> dict:
    """
    Avalia um conjunto de mapas gerados.

    Parâmetros
    ----------
    generated_maps : lista de np.array int8 [8, 8]
    training_maps  : lista de np.array do dataset de treino (para unseen_ratio)
    n_bins         : resolução do espaço de comportamento

    Retorna
    -------
    dict com todas as métricas
    """
    n = len(generated_maps)
    if n == 0:
        return {}

    # ── H1: Feasibility Ratio ────────────────────────────────────────────────
    feasible_flags = []
    for m in generated_maps:
        is_f, _ = feasibility_score(m)
        feasible_flags.append(is_f)
    feasibility_ratio = float(np.mean(feasible_flags))

    feasible_maps = [m for m, f in zip(generated_maps, feasible_flags) if f]

    # ── H2: Unique Ratio ─────────────────────────────────────────────────────
    tuples = [tuple(m.flatten()) for m in feasible_maps]
    unique_ratio = len(set(tuples)) / len(tuples) if tuples else 0.0

    # ── H2: Unseen Ratio ─────────────────────────────────────────────────────
    if training_maps is not None:
        train_set    = {tuple(m.flatten()) for m in training_maps}
        unseen_ratio = (sum(1 for t in tuples if t not in train_set) / len(tuples)
                        if tuples else 0.0)
    else:
        unseen_ratio = None

    # ── H3: BC Coverage ──────────────────────────────────────────────────────
    if feasible_maps:
        features_arr = np.array([compute_features(m) for m in feasible_maps])  # [n, 16]
        coverage_per_feature = []
        for fi in range(N_FEATURES):
            vals          = features_arr[:, fi]
            bins_occupied = len({int(v * (n_bins - 1)) for v in vals})
            coverage_per_feature.append(bins_occupied / n_bins)
        avg_coverage = float(np.mean(coverage_per_feature))
    else:
        features_arr         = None
        coverage_per_feature = [0.0] * N_FEATURES
        avg_coverage         = 0.0

    return {
        'feasibility_ratio'   : feasibility_ratio,
        'unique_ratio'        : unique_ratio,
        'unseen_ratio'        : float(unseen_ratio) if unseen_ratio is not None else None,
        'avg_bc_coverage'     : avg_coverage,
        'coverage_per_feature': coverage_per_feature,
        'n_generated'         : n,
        'n_feasible'          : len(feasible_maps),
        '_features_arr'       : features_arr,   # para ERA
    }


# ─────────────────────────────────────────────────────────────────────────────
# Relatório textual
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print("  RELATÓRIO DE AVALIAÇÃO — MLQD")
    print(sep)
    print(f"  Mapas gerados  : {results['n_generated']}")
    print(f"  Mapas viáveis  : {results['n_feasible']} "
          f"({results['feasibility_ratio']*100:.1f}%)   [H1]")
    print(f"  Unique ratio   : {results['unique_ratio']*100:.1f}%          [H2]")
    if results['unseen_ratio'] is not None:
        print(f"  Unseen ratio   : {results['unseen_ratio']*100:.1f}%          [H2]")
    print(f"  Cobertura BC ø : {results['avg_bc_coverage']*100:.1f}%          [H3]")
    print(f"\n  Cobertura por feature (H3):")
    for i, (name, cov) in enumerate(
            zip(FEATURE_NAMES, results['coverage_per_feature'])):
        bar = '█' * int(cov * 24)
        print(f"  F{i+1:02d} {name:<24} {bar:<24} {cov*100:5.1f}%")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Expressive Range Analysis  (H4)
# ─────────────────────────────────────────────────────────────────────────────

def expressive_range_plot(
        generated_maps: list,
        training_maps:  list | None = None,
        output_path:    str  | None = None,
) -> None:
    """
    Scatter 2D para 3 pares de features representativos (H4).
    Compara distribuição dos mapas gerados vs. dataset de treino.
    """
    if not generated_maps:
        print("[ERA] Nenhum mapa viável para plotar.")
        return

    gen_feats = np.array([compute_features(m) for m in generated_maps])

    # Pares: (F2 vs F9), (F10 vs F15), (F1 vs F4)
    pairs = [(1, 8), (9, 14), (0, 3)]
    names = [
        ('Wall Ratio (F2)',      'Graph Diameter (F9)'),
        ('Bases Distance (F10)', 'Exploration (F15)'),
        ('Floor Ratio (F1)',     'Horiz. Symmetry (F4)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Expressive Range Analysis', fontsize=13)

    for ax, (fi, fj), (nx, ny) in zip(axes, pairs, names):
        if training_maps is not None:
            tr_f = np.array([compute_features(m) for m in training_maps])
            ax.scatter(tr_f[:, fi], tr_f[:, fj],
                       alpha=0.25, s=6, c='gray', label='Treino', zorder=1)
        ax.scatter(gen_feats[:, fi], gen_feats[:, fj],
                   alpha=0.5, s=10, c='steelblue', label='Gerado', zorder=2)
        ax.set_xlabel(nx, fontsize=9)
        ax.set_ylabel(ny, fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[ERA] Gráfico salvo em '{output_path}'")
    else:
        plt.show()
    plt.close()

