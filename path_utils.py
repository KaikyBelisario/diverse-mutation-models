"""
path_utils.py — Gerenciamento Centralizado de Caminhos
-------------------------------------------------------
Define todos os caminhos de entrada/saída do pipeline em um único local.
Garante que a pasta 'resultados' existe.
"""

import os


# Pasta base do projeto
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pasta de resultados
RESULTS_DIR = os.path.join(PROJECT_DIR, "resultados")


def ensure_results_dir():
    """Cria a pasta 'resultados' se não existir."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# CAMINHOS DOS ARQUIVOS DE RESULTADO
# ============================================================================

def get_dataset_path():
    """Retorna o caminho do dataset de mutações do MAP-Elites."""
    return os.path.join(RESULTS_DIR, "dataset_mutacoes_qd.npy")


def get_qd_map_path():
    """Retorna o caminho do gráfico do espaço QD."""
    return os.path.join(RESULTS_DIR, "mapa_qd.png")


def get_mutation_model_path():
    """Retorna o caminho do modelo de mutação treinado."""
    return os.path.join(RESULTS_DIR, "modelo_mutacao.pth")


def get_sketch_model_path():
    """Retorna o caminho do modelo do gerador de esboços (legado — CNN condicional)."""
    return os.path.join(RESULTS_DIR, "gerador_esbocos.pth")


def get_transformer_model_path():
    """Retorna o caminho do Transformer treinado (MLQD §3.2)."""
    return os.path.join(RESULTS_DIR, "transformer_model.pth")


def get_era_plot_path():
    """Retorna o caminho do gráfico de Expressive Range Analysis."""
    return os.path.join(RESULTS_DIR, "era_plot.png")


def get_maps_output_path():
    """Retorna o caminho do arquivo de texto com os mapas gerados."""
    return os.path.join(RESULTS_DIR, "mapas_gerados.txt")


# ============================================================================
# FUNÇÃO DE VERIFICAÇÃO
# ============================================================================

def all_models_exist():
    """Retorna True se todos os modelos do pipeline revisado existem."""
    return (
        os.path.exists(get_dataset_path()) and
        os.path.exists(get_mutation_model_path()) and
        os.path.exists(get_transformer_model_path())
    )

