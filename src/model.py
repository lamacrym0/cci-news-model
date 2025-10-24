# model.py
import torch
import torch.nn as nn


class FeatureDropout(nn.Module):
    """Dropout sur les features (dernière dimension)"""
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
    LSTM profond avec residual connections entre couches.
    Utilise LSTM bidirectionnel pour capturer patterns avant/arrière.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 6, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projection initiale
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)
        
        # Couches LSTM empilées
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=hidden_dim * 2,  # Bidirectional input
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * 2))
            self.dropouts.append(nn.Dropout(dropout))

        # Projection finale (bidirectional = 2*hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = self.input_proj(x)  # (B, T, hidden_dim*2)
        
        for lstm, norm, drop in zip(self.layers, self.norms, self.dropouts):
            residual = x
            out, _ = lstm(x)  # (B, T, hidden_dim*2)
            out = norm(out)
            out = drop(out)
            # Residual connection (dimensions match now)
            x = out + residual
        
        x = self.output_proj(x)  # (B, T, hidden_dim)
        x = self.final_norm(x)
        return x


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head self-attention sur la séquence temporelle.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, 
            n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention avec residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN avec residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class CCILSTMRegressorBig(nn.Module):
    """
    Modèle LSTM profond pour prédiction ΔCCI :
      - Feature dropout en entrée
      - LSTM bidirectionnel profond (6 couches)
      - Multi-head attention temporelle
      - Pooling global + MLP head
    
    ~5M+ paramètres selon hidden_dim
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
        
        # Feature dropout
        self.fdrop = FeatureDropout(p=feature_dropout_p)
        
        # Encodeur LSTM profond
        self.encoder = DeepLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # Attention temporelle
        self.attention = MultiHeadTemporalAttention(
            hidden_dim=hidden_dim,
            n_heads=attn_heads,
            dropout=dropout
        )
        
        # Pooling temporel (moyenne sur toute la séquence)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP de sortie
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
        # x: (B, T, F)
        x = self.fdrop(x)              # Feature dropout
        x = self.encoder(x)            # (B, T, H)
        x = self.attention(x)          # (B, T, H)
        x = x.transpose(1, 2)          # (B, H, T)
        x = self.pool(x).squeeze(-1)   # (B, H)
        y = self.head(x)               # (B, 1)
        return y


def make_model(
    input_dim: int,
    sequence_length: int = 6,
    hidden_dim: int = 512,
    lstm_layers: int = 6,
    attn_heads: int = 8,
    dropout: float = 0.3,
    feature_dropout_p: float = 0.1
) -> nn.Module:
    """
    Factory pour modèle haute puissance.
    
    Args:
        input_dim: Nombre de features
        sequence_length: Longueur des séquences (pas utilisé dans le modèle)
        hidden_dim: Dimension cachée (défaut 512 pour ~5M params)
        lstm_layers: Nombre de couches LSTM (défaut 6)
        attn_heads: Nombre de têtes d'attention (défaut 8)
        dropout: Taux de dropout général
        feature_dropout_p: Taux de dropout sur les features
    
    Returns:
        Modèle LSTM profond initialisé
    """
    return CCILSTMRegressorBig(
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        feature_dropout_p=feature_dropout_p
    )