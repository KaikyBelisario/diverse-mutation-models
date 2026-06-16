# Estrutura de Diretórios — PCGML Pipeline

## Organização dos Arquivos

A partir da atualização, todos os arquivos **gerados** pelo pipeline são salvos na pasta `resultados/`.

### Estrutura

```
PCGML/
├── *.py                    # Scripts principais do pipeline
│   ├── main.py            # Orquestrador do pipeline completo
│   ├── map_elites.py      # Passo 1: Geração do dataset
│   ├── treino_mutacao.py  # Passo 2: Treinamento do modelo de mutação
│   ├── gerador_esbocos.py # Passo 3: Treinamento do gerador de esboços
│   ├── geracao_final.py   # Passo 4: Inferência final
│   ├── device_utils.py    # Detecção de GPU (AMD/NVIDIA) vs CPU
│   └── path_utils.py      # Centralização dos caminhos de arquivo
│
├── resultados/            # TODOS OS ARQUIVOS GERADOS VÃO AQUI
│   ├── dataset_mutacoes_qd.npy        # Dataset com transições evolutivas (Passo 1)
│   ├── mapa_qd.png                    # Gráfico do espaço QD (Passo 1)
│   ├── modelo_mutacao.pth             # Modelo treinado de mutação (Passo 2)
│   └── gerador_esbocos.pth            # Modelo treinado do gerador (Passo 3)
│
└── docs/                  # Documentação e artigos (input, não alterado)
    ├── context.txt
    ├── Diverse Level Generation via Machine Learning of Quality.pdf
    └── Mutation Models.pdf
```

## Como Usar

### Pipeline Completo (Passos 1–4)

```bash
python main.py
```

Isso irá:
1. Gerar o dataset MAP-Elites
2. Treinar o modelo de mutação
3. Treinar o gerador de esboços
4. Executar inferência com 3 perfis de teste

### Começar a Partir de um Passo Específico

Se os passos anteriores já foram executados, pule-os:

```bash
# Começa do Passo 2 (reusa o dataset do Passo 1)
python main.py --from 2

# Começa do Passo 3 (reusa o dataset e o modelo de mutação)
python main.py --from 3

# Começa do Passo 4 (usa todos os modelos já treinados)
python main.py --from 4
```

### Executar Apenas um Passo

```bash
# Apenas a geração do dataset
python main.py --only 1

# Apenas o treinamento do modelo de mutação
python main.py --only 2

# Apenas o treinamento do gerador
python main.py --only 3

# Apenas a inferência final
python main.py --only 4
```

## Importação via path_utils.py

Se você estiver **desenvolvendo scripts customizados**, importe as funções:

```python
from path_utils import (
    ensure_results_dir,
    get_dataset_path,
    get_qd_map_path,
    get_mutation_model_path,
    get_sketch_model_path,
    all_models_exist
)

# Garantir que a pasta existe
ensure_results_dir()

# Usar os caminhos
print(f"Dataset: {get_dataset_path()}")
print(f"Modelo de mutação: {get_mutation_model_path()}")

# Verificar se todos os modelos estão prontos
if all_models_exist():
    print("Sistema pronto para inferência!")
```

## GPU / Aceleração

### Detectar Hardware Disponível

O sistema seleciona automaticamente na sequência:

1. **GPU AMD** (ROCm) — se disponível
2. **GPU NVIDIA** (CUDA) — se disponível  
3. **CPU** — fallback

Para **ativar suporte real a GPU AMD**, instale:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

Após isso, o script exibirá `[AMD (ROCm)]` ao iniciar, confirmando que a GPU AMD foi detectada.

### Forçar CPU

Se quiser executar apenas em CPU (para testes):

```bash
CUDA_VISIBLE_DEVICES="" python main.py
```

## Limpeza

Para **recomeçar do zero** e deletar todos os resultados anteriores:

```bash
rm -rf resultados/
```

O script recriará a pasta automaticamente quando necessário.

