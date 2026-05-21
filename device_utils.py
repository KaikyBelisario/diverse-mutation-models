"""
device_utils.py — Seleção de Dispositivo Inteligente
-----------------------------------------------------
Ordem de prioridade:
  1. GPU AMD  (via PyTorch/ROCm  — `torch.cuda` com build ROCm)
  2. GPU NVIDIA (via PyTorch/CUDA — `torch.cuda` com build CUDA)
  3. CPU

NOTA: Em PyTorch, GPUs AMD com ROCm e GPUs NVIDIA com CUDA expõem
exactamente a mesma interface `torch.cuda`. A distinção entre
fabricantes é feita pelo nome do dispositivo reportado pelo driver.
Para usar uma GPU AMD instale o build ROCm do PyTorch:
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
"""

import torch

# Cache: imprime e detecta somente uma vez por sessão
_cached_device: "torch.device | None" = None


def get_device(verbose: bool = True) -> torch.device:
    """
    Retorna o melhor dispositivo disponível na seguinte ordem:
      AMD (ROCm) → NVIDIA (CUDA) → CPU

    Parâmetros
    ----------
    verbose : bool
        Se True, imprime qual dispositivo foi selecionado.

    Returns
    -------
    torch.device
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)

        # ROCm reporta nomes como "AMD Radeon ...", "Radeon RX ..." etc.
        is_amd = any(kw in gpu_name.upper() for kw in ("AMD", "RADEON", "INSTINCT", "VEGA"))

        if verbose:
            vendor = "AMD (ROCm)" if is_amd else "NVIDIA (CUDA)"
            print(f"[DEVICE] GPU selecionada: {gpu_name}  [{vendor}]")

        _cached_device = device
        return device

    if verbose:
        # Informa o usuário que GPUs AMD requerem o build ROCm do PyTorch
        print(
            "[DEVICE] Nenhuma GPU detectada pelo torch.cuda.\n"
            "         Para GPUs AMD instale: "
            "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2\n"
            "[DEVICE] Usando: CPU"
        )
    _cached_device = torch.device("cpu")
    return _cached_device


def device_info() -> str:
    """Retorna uma string descritiva do dispositivo ativo (útil para logs)."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"
