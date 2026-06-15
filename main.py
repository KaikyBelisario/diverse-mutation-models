"""
main.py — Pipeline Completo PCGML (Revisado)
=============================================
Implementa a fusão de MLQD (Sfikas et al., 2025) + Mutation Models (Khalifa et al., 2022).

Passos:
  1. CPA          → FI-CPA gera EHR com triplas (parent, loc, action)
  2. MutationCNN  → treina classificador de ações (Normal ou Assisted)
  3. Transformer  → treina gerador autoregressivo de mapas 8×8
  4. Inferência   → gera mapas, avalia H1–H4, plota ERA

Uso:
  python main.py                  # executa todos os passos
  python main.py --from 2         # começa do passo 2
  python main.py --only 4         # executa apenas o passo 4
  python main.py --assisted       # Passo 1 com Assisted Evolution
  python main.py --ehr 10000      # tamanho alvo do EHR (default: 50000)
"""

import os
# Deve ser definido antes de qualquer inicialização do ROCm/PyTorch
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys
import time
import argparse
from path_utils import ensure_results_dir

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEP = "=" * 64


def banner(titulo: str, passo: int) -> None:
    print(f"\n{SEP}\n  PASSO {passo}: {titulo}\n{SEP}")


def fmt_tempo(segundos: float) -> str:
    m, s = divmod(int(segundos), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Passo 1 — FI-CPA (gera EHR)
# ─────────────────────────────────────────────────────────────────────────────

def passo1(target_ehr: int = 50_000, assisted: bool = False) -> None:
    banner("FI-CPA — Gerando EHR de Mutações (MLQD §3.1 + §5.1.1)", 1)
    import numpy as np
    from cpa import FICPA
    from path_utils import get_dataset_path

    ensure_results_dir()
    t0 = time.time()

    retrain_cb = None
    retrain_iv = None

    if assisted:
        print("[Passo 1] Modo Assisted Evolution ativado (Khalifa et al. §3.2)")
        from treino_mutacao import assisted_retrain_callback
        retrain_cb = assisted_retrain_callback
        retrain_iv = max(500, target_ehr // 20)   # re-treina ~20 vezes

    cpa = FICPA(
        target_ehr       = target_ehr,
        retrain_interval = retrain_iv,
        retrain_callback = retrain_cb,
    )
    ehr = cpa.run()

    np.save(get_dataset_path(), ehr)
    print(f"-> EHR salvo: {len(ehr):,} entradas em '{get_dataset_path()}'")
    print(f"-> Células viáveis nos arquivos: {cpa.feasible_cell_count}")
    print(f"-> Tempo: {fmt_tempo(time.time() - t0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Passo 2 — Treino do MutationCNN
# ─────────────────────────────────────────────────────────────────────────────

def passo2() -> None:
    banner("MutationCNN — Imitando Mutações Evolutivas (Khalifa et al. §3)", 2)
    t0 = time.time()
    from treino_mutacao import train_mutation_model
    train_mutation_model(epochs=8)   # método Normal — treino único no EHR final
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    print(f"-> Tempo: {fmt_tempo(time.time() - t0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Passo 3 — Treino do Transformer
# ─────────────────────────────────────────────────────────────────────────────

def passo3() -> None:
    banner("Transformer — Aprendendo Distribuição QD (MLQD §3.2)", 3)
    t0 = time.time()
    from transformer_model import train_transformer
    train_transformer()
    print(f"-> Tempo: {fmt_tempo(time.time() - t0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Passo 4 — Inferência Final + Avaliação
# ─────────────────────────────────────────────────────────────────────────────

def passo4() -> None:
    banner("Inferência + Avaliação (MLQD §5.2 H1–H4)", 4)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")

    from geracao_final import generate_map, load_transformer, load_mutation_cnn
    from avaliacao import evaluate_generated_maps, print_report, expressive_range_plot
    from path_utils import get_dataset_path, get_era_plot_path
    from device_utils import get_device

    device = get_device()

    try:
        transformer  = load_transformer(device)
        mutation_cnn = load_mutation_cnn(device)
    except FileNotFoundError as e:
        print(f"[ERRO] {e}\nExecute os passos anteriores primeiro.")
        return

    # ── Gera 200 mapas para avaliação ────────────────────────────────────────
    N_EVAL = 200
    print(f"\n[Avaliação] Gerando {N_EVAL} mapas para avaliação...")
    generated = []
    for i in range(N_EVAL):
        m = generate_map(mutation_steps=3, verbose=False,
                         _transformer=transformer, _mutation_cnn=mutation_cnn)
        generated.append(m)
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{N_EVAL} mapas gerados")

    # ── Carrega amostra do dataset de treino para unseen_ratio ────────────────
    training_sample = None
    try:
        ehr = np.load(get_dataset_path(), allow_pickle=True).tolist()
        training_sample = [np.array(item['child'], dtype=np.int8) for item in ehr[:5000]]
    except FileNotFoundError:
        pass

    # ── Métricas ──────────────────────────────────────────────────────────────
    results = evaluate_generated_maps(generated, training_maps=training_sample)
    print_report(results)

    # ── Expressive Range Analysis (H4) ────────────────────────────────────────
    feasible_gen = [m for m in generated]   # ERA sobre todos (viáveis filtrados internamente)
    expressive_range_plot(
        generated_maps = feasible_gen,
        training_maps  = training_sample,
        output_path    = get_era_plot_path(),
    )

    # ── Salva todos os mapas em arquivo ─────────────────────────────────────────
    from geracao_final import format_level
    from path_utils import get_maps_output_path
    maps_path = get_maps_output_path()
    with open(maps_path, 'w', encoding='utf-8') as f:
        f.write(f"Mapas Gerados — {N_EVAL} mapas\n")
        f.write("=" * 44 + "\n")
        for i, m in enumerate(generated):
            f.write(format_level(m, f"Mapa {i+1}/{N_EVAL}"))
            f.write("\n")
    print(f"\n[Exemplos] {N_EVAL} mapas salvos em '{maps_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Entrada Principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline PCGML — MLQD + Mutation Models")
    parser.add_argument("--from",     dest="start_from", type=int, default=1,
                        help="Começa a partir do passo N")
    parser.add_argument("--only",     dest="only_step",  type=int, default=None,
                        help="Executa apenas o passo N")
    parser.add_argument("--assisted", action="store_true",
                        help="Passo 1 com Assisted Evolution")
    parser.add_argument("--ehr",      dest="ehr_size",   type=int, default=50_000,
                        help="Tamanho alvo do EHR (default: 50000)")
    args = parser.parse_args()

    ensure_results_dir()

    passos = {
        1: lambda: passo1(target_ehr=args.ehr_size, assisted=args.assisted),
        2: passo2,
        3: passo3,
        4: passo4,
    }

    t_total = time.time()
    print(f"\n{SEP}")
    print("  PCGML — MLQD + Mutation Models Pipeline")
    print(f"  EHR alvo: {args.ehr_size:,} | Assisted: {args.assisted}")
    print(SEP)

    if args.only_step:
        if args.only_step not in passos:
            print(f"[ERRO] Passo inválido: {args.only_step}. Use 1–4.")
            sys.exit(1)
        passos[args.only_step]()
    else:
        for num, fn in passos.items():
            if num >= args.start_from:
                fn()

    print(f"\n{SEP}")
    print(f"  Pipeline concluído! Tempo total: {fmt_tempo(time.time() - t_total)}")
    print(SEP)
