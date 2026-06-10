# Contexto do Projeto — PCGML: Geração de Níveis por Fusão de MLQD e Mutation Models

## Visão Geral

Este projeto é uma implementação prática desenvolvida como parte de uma monografia sobre **geração procedural de conteúdo em videogames com Inteligência Artificial (PCG — Procedural Content Generation)**. O código representa a fusão de duas metodologias publicadas em conferências científicas da área de jogos digitais:

1. **MLQD — Machine Learning of Quality Diversity** (Sfikas, Liapis & Yannakakis, FDG 2025)
2. **Mutation Models: Learning to Generate Levels by Imitating Evolution** (Khalifa, Green & Togelius, FDG 2022)

O objetivo central é criar um gerador de mapas de estratégia 8×8 que seja **rápido o suficiente para uso em tempo de execução**, **capaz de satisfazer restrições funcionais** e produza **saídas diversas** — combinando as vantagens dos geradores construtivos (velocidade) com as dos geradores baseados em busca (qualidade e controlabilidade).

---

## Domínio: Mapas de Estratégia 8×8

O domínio escolhido para validação são **esboços de mapas para jogos de estratégia** em grade 8×8, estabelecidos na literatura como banco de testes padrão para PCG com restrições.

### Representação
Cada mapa é uma matriz `int8` de dimensão 8×8, onde cada célula assume um dos quatro tipos de tile:

| Valor | Tipo     | Descrição                                  |
|-------|----------|--------------------------------------------|
| 0     | Floor    | Tile transponível, sem função especial     |
| 1     | Wall     | Obstáculo; bloqueia movimento e pathfinding|
| 2     | Resource | Tile de recurso (tile especial)            |
| 3     | Base     | Base de jogador (tile especial)            |

### Restrições de Viabilidade
Um mapa é considerado **viável** se satisfaz três condições simultaneamente:
- **(a)** Exatamente **2 bases**
- **(b)** Entre **4 e 10 recursos**
- **(c)** Todos os tiles especiais (bases e recursos) estão **conectados** por um caminho livre de paredes

Um score de inviabilidade contínuo `f_inf ∈ [0, 1]` quantifica o quão próximo um mapa inviável está de ser corrigido (Eq. 1 de Sfikas et al.):

$$F_{inf} = \frac{1}{2}\frac{c_b}{b \cdot (b-1)} + \frac{1}{2}\frac{c_r}{r \cdot b}$$

onde $b$ e $r$ são o número de bases e recursos, e $c_b$, $c_r$ são os pares conectados entre bases e entre bases-recursos, respectivamente.

### Espaço de Comportamento (16 Features)
Para medir e maximizar a diversidade dos mapas gerados, são calculadas **16 métricas de comportamento** (`F1`–`F16`):

- **F1–F3**: Proporção de Floor, Wall e Resource
- **F4–F7**: Simetria horizontal, vertical, diagonal e anti-diagonal
- **F8**: Número de ilhas de Wall (componentes isolados)
- **F9**: Diâmetro do grafo de conectividade (maior distância entre tiles passáveis)
- **F10**: Distância entre as duas bases
- **F11–F12**: Segurança dos recursos (distância mínima até base) e seu balanceamento
- **F13–F14**: Segurança das bases (distância mínima até recurso) e seu balanceamento
- **F15–F16**: Exploração do mapa (cobertura média de pathfinding) e seu balanceamento

---

## Artigo 1 — Mutation Models (Khalifa et al., FDG 2022)

### Problema Abordado
Geradores baseados em busca evolutiva garantem qualidade, mas são lentos demais para uso em tempo real. Geradores construtivos são rápidos, mas difíceis de especificar com garantias de viabilidade.

### Proposta
Treinar uma rede neural para **imitar as mutações bem-sucedidas da evolução**. O modelo aprende, a partir do histórico evolutivo, quais modificações locais melhoram um mapa, eliminando a necessidade de função de fitness durante a inferência.

### Componentes Principais

**Loop Evolutivo (`μ+λ`):**
Evolução padrão que registra o **histórico de cada mutação** — a sequência de `(estado do mapa, localização, ação)` que levou cada cromossomo ao seu estado atual.

**Registro de Histórico Evolutivo (EHR — Evolutionary History Record):**
Estrutura de dados que acumula **triplas** `{parent, loc_r, loc_c, action}` de todas as mutações bem-sucedidas ao longo da evolução.

**MutationCNN:**
CNN treinada em modo supervisionado sobre o EHR. Usa a **representação Narrow** (inspirada em PCGRL): o input é uma janela 8×8 recortada ao redor do ponto de mutação, e o output é uma classificação em 4 ações: `{NoChange=0, Floor=1, Wall=2, Resource=3}`.

**Dois modos de treinamento:**
- **Normal**: treina a CNN uma única vez ao final da evolução, sobre o EHR completo.
- **Assisted**: re-treina a CNN a cada `N` passos durante a evolução e a utiliza como operador de mutação para os passos seguintes — análogo ao aprendizado por reforço on-policy.

---

## Artigo 2 — MLQD (Sfikas et al., FDG 2025)

### Problema Abordado
PCGML requer grandes datasets de conteúdo existente (frequentemente indisponível ou protegido por direitos autorais). Algoritmos QD geram conteúdo diverso, mas sofrem com a "maldição da dimensionalidade" em espaços de alta dimensão.

### Proposta
**MLQD (Machine Learning of Quality Diversity)**: pipeline auto-suficiente em dois passos — (1) um algoritmo evolutivo QD gera um dataset grande e diverso do zero; (2) um Transformer aprende a distribuição do dataset e passa a emular o comportamento QD via inferência estocástica.

### Componente 1 — CPA (Cross-Pollination of Axis-Aligned Archives)

O CPA mantém **N arquivos 1D independentes**, um para cada feature do espaço de comportamento. Cada arquivo é dividido em `N_BINS` células (bins). O mecanismo de "cross-pollination" troca indivíduos entre arquivos durante a evolução, indiretamente cobrindo todo o espaço multidimensional de features sem o custo computacional de avaliá-las todas simultaneamente.

**Fases de operação:**
1. **Inicialização**: mapas são gerados aleatoriamente (com viabilidade garantida) e inseridos nos arquivos conforme seus valores de feature.
2. **Operação principal**: um arquivo pai é selecionado aleatoriamente; um mapa é amostrado desse arquivo como progenitor; o mapa sofre mutação pontual (1 tile); o filho é inserido em um arquivo sorteado aleatoriamente com base em seu valor naquela feature.
3. **EHR**: todos os indivíduos gerados ao longo da evolução são armazenados — não apenas a população final.

**FI-CPA (Feasible-Infeasible CPA):** variante implementada neste projeto que mantém um arquivo adicional para indivíduos inviáveis, guiados pelo score `f_inf` para gradualmente convergirem à viabilidade.

### Componente 2 — Transformer (Gerador Autoregressivo)

Arquitetura **GPT-style** (decoder-only com máscara causal):

| Hiperparâmetro      | Valor |
|---------------------|-------|
| `vocab_size`        | 6 (4 tiles + token neutro + token máscara) |
| `seq_len`           | 65 (1 token neutro + 64 tiles achatados em row-major) |
| `embed_dim`         | 256 |
| `n_layers`          | 2 |
| `n_heads`           | 2 |
| `ff_dim`            | 256 |
| Splits de treino    | 80% / 15% / 5% (treino/validação/teste) |
| Early stopping      | patience = 3 |
| Inferência          | top-p sampling, p = 0.9 |

O mapa 8×8 é serializado como uma sequência de 65 tokens (row-major) prefixada por um token neutro. Na inferência, o Transformer gera token por token de forma autoregressiva até completar os 64 tiles.

---

## Fusão das Abordagens

Este projeto combina as duas metodologias em um **pipeline de 4 passos**:

```
┌─────────────────────────────────────────────────────────────────┐
│  PASSO 1: FI-CPA  (cpa.py)                                      │
│  Gera EHR com triplas {parent, loc_r, loc_c, action}            │
│  Opcionalmente: Assisted Evolution com MutationCNN como mutador │
├─────────────────────────────────────────────────────────────────┤
│  PASSO 2: MutationCNN  (treino_mutacao.py)                      │
│  CNN treinada sobre o EHR para classificar ações locais         │
│  (imita mutações evolutivas bem-sucedidas — Khalifa et al.)     │
├─────────────────────────────────────────────────────────────────┤
│  PASSO 3: Transformer  (transformer_model.py)                   │
│  Treinado sobre os mapas filhos viáveis do EHR                  │
│  (aprende distribuição QD do dataset — Sfikas et al.)           │
├─────────────────────────────────────────────────────────────────┤
│  PASSO 4: Inferência + Avaliação  (geracao_final.py, avaliacao) │
│  Transformer gera mapa globalmente; MutationCNN refina via      │
│  scanline (narrow representation). Nenhuma fitness function.    │
└─────────────────────────────────────────────────────────────────┘
```

### Lógica da Fusão

O ponto central da fusão é que o **EHR gerado pelo CPA serve simultaneamente como dataset para dois modelos distintos**:

- O **Transformer** aprende a distribuição global dos mapas viáveis, capturando a diversidade produzida pelo CPA (contribuição MLQD).
- A **MutationCNN** aprende as correções locais que a evolução realizou para melhorar os mapas, funcionando como um refinador em tempo de inferência (contribuição Mutation Models).

Na geração final, os dois modelos atuam em sequência:
1. O Transformer produz um mapa esboçado com boa coerência global.
2. A MutationCNN passa por cada posição do mapa (scanline) e aplica correções locais sem recorrer a nenhuma função de fitness.

O modo **Assisted Evolution** aprofunda ainda mais a fusão: durante a própria etapa de coleta de dados (Passo 1), a MutationCNN já treinada guia as mutações do CPA, criando um ciclo de melhoria análogo ao aprendizado on-policy — melhorando simultaneamente a qualidade do dataset e a capacidade do modelo de mutação.

---

## Avaliação

As hipóteses de avaliação (`avaliacao.py`) seguem o protocolo de Sfikas et al. (§5.2):

| Hipótese | Métrica | Descrição |
|----------|---------|-----------|
| H1 | Feasibility Ratio | Proporção de mapas gerados que satisfazem as três restrições |
| H2 | Unique Ratio + Unseen Ratio | Diversidade interna e proporção de mapas não vistos no treino |
| H3 | Coverage | Percentual de bins do espaço de comportamento cobertos pelos mapas gerados |
| H4 | ERA (Expressive Range Analysis) | Scatter 2D de features para visualizar a dispersão no espaço de comportamento |
| H5 | Latência | Velocidade de geração em relação ao processo evolutivo original |

---

## Estrutura de Arquivos

| Arquivo | Responsabilidade |
|---------|-----------------|
| `main.py` | Ponto de entrada; orquestra os 4 passos via argparse |
| `domain.py` | Representação, restrições, mutação e 16 features dos mapas |
| `cpa.py` | Algoritmo FI-CPA; geração e armazenamento do EHR |
| `treino_mutacao.py` | MutationCNN: dataset, arquitetura, treino Normal/Assisted |
| `transformer_model.py` | Transformer decoder-only: tokenização, dataset, treino, inferência |
| `geracao_final.py` | Pipeline de inferência: Transformer → MutationCNN → mapa final |
| `avaliacao.py` | Métricas H1–H5 e plotagem de ERA |
| `device_utils.py` | Seleção automática de dispositivo (CUDA / ROCm / CPU) |
| `path_utils.py` | Centraliza caminhos de artefatos em `resultados/` |
| `runtime_config.py` | Configurações de execução ajustáveis |

### Artefatos Gerados (`resultados/`)

| Arquivo | Conteúdo |
|---------|----------|
| `dataset_mutacoes_qd.npy` | EHR serializado (array de dicts numpy) |
| `modelo_mutacao.pth` | Pesos do MutationCNN treinado |
| `transformer_model.pth` | Pesos do Transformer treinado |
| `era_plot.png` | Experience Range Analisys |

---

## Referências

- Sfikas, K., Liapis, A., & Yannakakis, G. N. (2025). *Diverse Level Generation via Machine Learning of Quality Diversity*. FDG '25, Graz, Austria. ACM. https://doi.org/10.1145/3723498
- Khalifa, A., Green, M. C., & Togelius, J. (2022). *Mutation Models: Learning to Generate Levels by Imitating Evolution*. FDG '22, Athens, Greece. ACM. https://doi.org/10.1145/3555858
