# model.py (version HAUTE PUISSANCE)
import math
import torch
import torch.nn as nn


class FeatureDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        mask = torch.empty(x.shape[-1], device=x.device).bernoulli_(1 - self.p)
        return x * mask


class DeepLSTMEncoder(nn.Module):
    """
    LSTM très profond, large, avec residual connections entre couches.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 6, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    hidden_size=hidden_dim,
                    input_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * 2))
            self.dropouts.append(nn.Dropout(dropout))

        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for lstm, norm, drop in zip(self.layers, self.norms, self.dropouts):
            out, _ = lstm(x)
            out = norm(out)
            out = drop(out)
            x = out + x  # Residual connection
        x = self.output_proj(x)
        x = self.final_norm(x)
        return x


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention sur la séquence (comme Transformer).
    """
    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm(x + self.dropout(ffn_out))
        return x


class CCILSTMRegressorBig(nn.Module):
    """
    Modèle GROS pour ressources illimitées :
      - LSTM très profond (6 couches)
      - Multi-head attention
      - Residual connections
      - ~5M+ paramètres
    """
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 6,
        hidden_dim: int = 512,
        lstm_layers: int = 6,
        attn_heads: int = 8,
        dropout: float = 0.3,
        feature_dropout_p: float = 0.1
    ):
        super().__init__()
        self.fdrop = FeatureDropout(p=feature_dropout_p)
        self.encoder = DeepLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        self.attention = MultiHeadTemporalAttention(
            hidden_dim=hidden_dim,
            n_heads=attn_heads,
            dropout=dropout
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average over time
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fdrop(x)
        x = self.encoder(x)           # (B, T, H)
        x = self.attention(x)         # (B, T, H)
        x = x.transpose(1, 2)         # (B, H, T)
        x = self.pool(x).squeeze(-1)  # (B, H)
        y = self.head(x)
        return y


def make_model_big(
    input_dim: int,
    sequence_length: int = 6
) -> nn.Module:
    """
    Pour très gros calcul :
      - 5M+ paramètres
      - Utilise 24–48 Go VRAM en FP16
      - Idéal pour A100 80GB ou multi-GPU
    """
    return CCILSTMRegressorBig(
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_dim=512,
        lstm_layers=6,
        attn_heads=8,
        dropout=0.3,
        feature_dropout_p=0.1
    )