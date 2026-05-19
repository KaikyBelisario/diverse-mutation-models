"""
main.py — Pipeline Completo PCGML
Executa os 4 passos em ordem:
  Passo 1: MAP-Elites         → gera o dataset de transições evolutivas
  Passo 2: Modelo de Mutação  → treina a CNN residual (pai → filho)
  Passo 3: Gerador de Esboços → treina o gerador condicional (BC1, BC2 → mapa)
  Passo 4: Inferência Final   → gera mapas com os parâmetros desejados
"""

import os
import sys
import time
from path_utils import ensure_results_dir

# Garante que o diretório de trabalho é o da pasta do projeto
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEPARATOR = "=" * 64


def banner(titulo, passo):
    print(f"\n{SEPARATOR}")
    print(f"  PASSO {passo}: {titulo}")
    print(f"{SEPARATOR}")


def tempo(segundos):
    m, s = divmod(int(segundos), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# PASSO 1 — MAP-Elites
# ---------------------------------------------------------------------------
def passo1():
    banner("MAP-Elites — Gerando Dataset de Mutações", 1)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")           # Backend sem janela, não bloqueia o pipeline
    import matplotlib.pyplot as plt
    from map_elites import MAPElites
    from path_utils import get_dataset_path, get_qd_map_path

    ensure_results_dir()

    t0 = time.time()
    optimizer = MAPElites()
    dataset = optimizer.run()
    np.save(get_dataset_path(), dataset)
    print(f"-> Dataset salvo: {len(dataset)} transições em '{get_dataset_path()}'")

    plt.figure(figsize=(6, 5))
    plt.imshow(optimizer.archive_fitness, origin="lower", cmap="viridis")
    plt.colorbar(label="Fitness")
    plt.title("Espaço de Qualidade-Diversidade")
    plt.savefig(get_qd_map_path(), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"-> Gráfico QD salvo em '{get_qd_map_path()}'")
    print(f"-> Tempo: {tempo(time.time() - t0)}")


# ---------------------------------------------------------------------------
# PASSO 2 — Treino do Modelo de Mutação
# ---------------------------------------------------------------------------
def passo2():
    banner("Treinamento do Modelo de Mutação (CNN Residual)", 2)
    t0 = time.time()
    from treino_mutacao import train_model
    train_model()
    print(f"-> Tempo: {tempo(time.time() - t0)}")


# ---------------------------------------------------------------------------
# PASSO 3 — Treino do Gerador de Esboços
# ---------------------------------------------------------------------------
def passo3():
    banner("Treinamento do Gerador de Esboços Condicional", 3)
    t0 = time.time()
    from gerador_esbocos import train_sketch_generator
    train_sketch_generator()
    print(f"-> Tempo: {tempo(time.time() - t0)}")


# ---------------------------------------------------------------------------
# PASSO 4 — Inferência Final
# ---------------------------------------------------------------------------
def passo4():
    banner("Inferência Final — Gerando Labirintos Personalizados", 4)
    from geracao_final import generate_custom_level

    # Testa três perfis distintos para confirmar que os parâmetros têm efeito
    perfis = [
        {"nome": "Aberto / Simples",     "density": 0.10, "complexity": 0.20},
        {"nome": "Equilibrado",          "density": 0.30, "complexity": 0.50},
        {"nome": "Denso / Complexo",     "density": 0.55, "complexity": 0.75},
    ]

    for p in perfis:
        print(f"\n{'─'*50}")
        print(f"  Perfil: {p['nome']}")
        generate_custom_level(
            target_density=p["density"],
            target_complexity=p["complexity"],
            mutation_steps=4
        )


# ---------------------------------------------------------------------------
# ENTRADA PRINCIPAL
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Permite pular passos já concluídos via argumento:
    # python main.py --from 2   (começa do passo 2)
    # python main.py --only 4   (executa apenas o passo 4)

    ensure_results_dir()

    start_from = 1
    only_step = None

    args = sys.argv[1:]
    if "--from" in args:
        start_from = int(args[args.index("--from") + 1])
    if "--only" in args:
        only_step = int(args[args.index("--only") + 1])

    passos = {1: passo1, 2: passo2, 3: passo3, 4: passo4}

    inicio_total = time.time()
    print("\n" + SEPARATOR)
    print("  PCGML — Pipeline de Geração Procedural de Labirintos")
    print(SEPARATOR)

    if only_step:
        if only_step not in passos:
            print(f"[ERRO] Passo inválido: {only_step}. Use 1, 2, 3 ou 4.")
            sys.exit(1)
        passos[only_step]()
    else:
        for num, fn in passos.items():
            if num >= start_from:
                fn()

    print(f"\n{SEPARATOR}")
    print(f"  Pipeline concluído! Tempo total: {tempo(time.time() - inicio_total)}")
    print(SEPARATOR)

