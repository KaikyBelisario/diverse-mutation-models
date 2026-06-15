"""
transformer_model.py — Transformer Decoder-Only para Geração de Mapas
----------------------------------------------------------------------
Implementação fiel ao MLQD §3.2 e §4.2 (Sfikas et al., 2025):

  - Arquitetura GPT-style (decoder-only com causal mask)
  - vocab_size = 6  (4 tile types + token neutro + token máscara)
  - seq_len    = 65 (1 token neutro + 64 tiles achatados em row-major)
  - embed_dim  = 256, n_layers = 2, n_heads = 2, ff_dim = 256  (§4.2.1)
  - Treino: cross-entropy; splits 80/15/5; early stopping patience=3
  - Inferência: autoregressiva com top-p sampling  p=0.9  (§4.2.4)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from device_utils import get_device
from path_utils import ensure_results_dir, get_dataset_path, get_transformer_model_path

# ── Constantes de tokenização (§4.2.2) ────────────────────────────────────────
MASK_TOKEN    = 0   # reservado para mascaramento na inferência
NEUTRAL_TOKEN = 1   # token inicial da sequência
# tile 0 (Floor) → token 2, tile 1 (Wall) → 3, tile 2 (Resource) → 4, tile 3 (Base) → 5
TOKEN_OFFSET  = 2

VOCAB_SIZE = 6
SEQ_LEN    = 65   # 1 neutro + 64 tiles
GRID_SIZE  = 8


# ─────────────────────────────────────────────────────────────────────────────
# Funções de tokenização
# ─────────────────────────────────────────────────────────────────────────────

def map_to_tokens(level_2d) -> np.ndarray:
    """Converte mapa 8×8 (int, valores 0-3) → sequência de 65 tokens (int64)."""
    flat   = np.asarray(level_2d, dtype=np.int64).flatten()   # [64]
    tokens = flat + TOKEN_OFFSET                               # [64] valores 2-5
    return np.concatenate([[NEUTRAL_TOKEN], tokens])           # [65]


def tokens_to_map(tokens) -> np.ndarray:
    """Converte sequência de 65 tokens → mapa 8×8 (int8)."""
    tile_tokens = np.asarray(tokens)[1:65]
    tiles = np.clip(tile_tokens - TOKEN_OFFSET, 0, 3).astype(np.int8)
    return tiles.reshape(GRID_SIZE, GRID_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TransformerMapDataset(Dataset):
    """
    Carrega o EHR do CPA e extrai sequências de tokens dos mapas filho (viáveis).
    Cada exemplo: (input_seq [64 tokens], target_seq [64 tokens]) para
    previsão next-token (input=tokens[0:64], target=tokens[1:65]).
    """
    def __init__(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        seqs = [map_to_tokens(item['child']) for item in data]

        # Remove duplicatas (§4.2.3 — shuffle ocorre no DataLoader)
        unique = list({tuple(s) for s in seqs})
        self._sequences = np.array(unique, dtype=np.int64)
        print(f"[TransformerDS] {len(data):,} entradas → {len(self._sequences):,} únicas")

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx):
        seq = self._sequences[idx]
        return (torch.tensor(seq[:-1], dtype=torch.long),
                torch.tensor(seq[1:],  dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Arquitetura: Transformer Decoder-Only  (GPT-style)
# ─────────────────────────────────────────────────────────────────────────────

class MapTransformer(nn.Module):
    """
    Transformer decoder-only para geração autoregressiva de sequências de tiles.

    Token embedding + positional embedding → N camadas de TransformerEncoder
    com causal mask → projeção linear → logits sobre VOCAB_SIZE.

    Usar TransformerEncoderLayer com causal mask é equivalente ao decoder
    self-attention e é a implementação padrão de modelos GPT-style no PyTorch.
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 seq_len:    int = SEQ_LEN,
                 embed_dim:  int = 256,
                 n_heads:    int = 2,
                 n_layers:   int = 2,
                 ff_dim:     int = 256,
                 dropout:    float = 0.1):
        super().__init__()

        self.seq_len   = seq_len
        self.embed_dim = embed_dim

        # Embeddings de token e posição (§3.2)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(seq_len, embed_dim)
        self.drop      = nn.Dropout(dropout)

        # Pilha de camadas de Transformer (decoder-only via causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,          # Pre-LN: mais estável para treino
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projeção final para logits
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Máscara causal (triangular inferior) — fixada em buffer
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(seq_len)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T]  — sequência de tokens (T ≤ seq_len)
        → logits [B, T, vocab_size]
        """
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)          # [1, T]
        emb = self.drop(self.token_emb(x) + self.pos_emb(pos))       # [B, T, D]

        # Máscara causal: impede o modelo de ver tokens futuros
        mask = self.causal_mask[:T, :T]                               # [T, T]
        out  = self.transformer(emb, mask=mask,
                                is_causal=True)                       # [B, T, D]
        return self.output_proj(out)                                  # [B, T, V]

    @torch.no_grad()
    def generate(self, device: torch.device, top_p: float = 0.9) -> np.ndarray:
        """
        Gera um mapa completo via amostragem autoregressiva com top-p (nucleus).
        Retorna np.array int8 [8, 8]  (§4.2.4).
        """
        self.eval()
        tokens = [NEUTRAL_TOKEN]

        for _ in range(SEQ_LEN - 1):
            inp    = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self(inp)                    # [1, T, V]
            probs  = F.softmax(logits[0, -1], dim=-1)

            # Nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            # Remove tokens que ultrapassam o limiar p
            cutoff = (cumulative - sorted_probs) > top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs /= sorted_probs.sum().clamp(min=1e-9)

            sampled    = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx[sampled].item()
            tokens.append(next_token)

        return tokens_to_map(np.array(tokens))


# ─────────────────────────────────────────────────────────────────────────────
# Loop de Treinamento
# ─────────────────────────────────────────────────────────────────────────────

def train_transformer():
    """
    Treina o MapTransformer no dataset gerado pelo CPA.
    Salva o melhor modelo (menor val loss) com early stopping (patience=3).
    Splits: 80% treino / 15% validação / 5% teste  (MLQD §4.2.3).
    """
    ensure_results_dir()
    device = get_device()

    print("[Transformer] Carregando dataset do EHR...")
    try:
        full_ds = TransformerMapDataset(get_dataset_path())
    except FileNotFoundError:
        print(f"Erro: '{get_dataset_path()}' não encontrado. Execute o Passo 1 primeiro!")
        return

    n       = len(full_ds)
    n_train = int(0.80 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, _ = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
    print(f"[Transformer] Train={n_train:,} | Val={n_val:,} | Test={n_test:,}")

    model     = MapTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=MASK_TOKEN)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Transformer] Parâmetros treináveis: {n_params:,}")

    best_val  = float('inf')
    no_improv = 0
    patience  = 3

    for epoch in range(50):
        # ── Treino ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)                                # [B, T, V]
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        # ── Validação ────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Época [{epoch+1:>2}/50] | "
                  f"Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

        # Early stopping
        if avg_v < best_val:
            best_val  = avg_v
            no_improv = 0
            torch.save(model.state_dict(), get_transformer_model_path())
        else:
            no_improv += 1
            if no_improv >= patience:
                print(f"   Early stopping na época {epoch + 1}.")
                break

    print(f"[Transformer] Concluído! Melhor val loss: {best_val:.4f}")
    print(f"[Transformer] Modelo salvo em '{get_transformer_model_path()}'.")


if __name__ == "__main__":
    train_transformer()

