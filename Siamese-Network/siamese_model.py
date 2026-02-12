"""
Siamese Neural Network for Trajectory Fragment Similarity Learning

Architecture:
- Twin LSTM encoders (shared weights) to process variable-length trajectory sequences
- Produces fixed-size embeddings for each fragment
- Similarity head computes probability that fragments belong to same vehicle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TrajectoryEncoder(nn.Module):
    """
    LSTM-based encoder for trajectory fragments

    Processes variable-length sequences of [x, y, velocity, time]
    and produces fixed-size embedding vectors
    """

    def __init__(self,
                 input_size: int = 4,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        Args:
            input_size: Number of features per timestep (default: 4 for x, y, v, t)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(TrajectoryEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output dimension
        self.output_size = hidden_size * self.num_directions

        # Optional: Add layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM encoder

        Args:
            sequences: Padded sequences of shape (batch, max_seq_len, input_size)
            lengths: Actual sequence lengths of shape (batch,)

        Returns:
            embeddings: Fixed-size embeddings of shape (batch, output_size)
        """
        batch_size = sequences.size(0)

        # Pack padded sequences for efficient LSTM processing
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_sequences)

        # 1. Global Average Pooling over LSTM outputs (Recommended for trajectory behavior)
        # Unpack the sequences back to (batch, max_seq_len, hidden_size * num_directions)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Create mask for valid timesteps
        mask = torch.arange(output.size(1)).expand(output.size(0), output.size(1)).to(output.device)
        mask = mask < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        # Average pooling with mask
        embedding = (output * mask).sum(dim=1) / lengths.unsqueeze(1).float()

        # Apply layer normalization
        embedding = self.layer_norm(embedding)

        return embedding


class SiameseTrajectoryNetwork(nn.Module):
    """
    Siamese Network for learning trajectory fragment similarity

    Architecture:
    1. Twin LSTM encoders (shared weights) process each fragment
    2. Symmetric feature combination (abs diff and product)
    3. Similarity head computes probability of same vehicle

    Enhanced version accepts endpoint features (time_gap, x_gap, y_gap, velocity_diff)
    that are concatenated with the symmetric features before the similarity head.
    """

    def __init__(self,
                 input_size: int = 4,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 similarity_hidden_dim: int = 64,
                 endpoint_feature_dim: int = 4,
                 use_endpoint_features: bool = True):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            similarity_hidden_dim: Hidden dimension for similarity head
            endpoint_feature_dim: Dimension of endpoint features (default: 4)
            use_endpoint_features: Whether to use endpoint features in similarity head
        """
        super(SiameseTrajectoryNetwork, self).__init__()

        self.use_endpoint_features = use_endpoint_features
        self.endpoint_feature_dim = endpoint_feature_dim

        # Shared trajectory encoder
        self.encoder = TrajectoryEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        embedding_size = self.encoder.output_size

        # Symmetric features: |a - b| and (a * b)
        # Each has the same dimension as the embedding
        if use_endpoint_features:
            combined_size = embedding_size * 2 + endpoint_feature_dim
        else:
            combined_size = embedding_size * 2

        # Enhanced similarity head with more capacity
        self.similarity_head = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_one(self, sequence: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """
        Encode a single fragment

        Args:
            sequence: Padded sequence (batch, seq_len, input_size)
            length: Actual lengths (batch,)

        Returns:
            embedding: (batch, embedding_size)
        """
        return self.encoder(sequence, length)

    def forward(self,
                seq_a: torch.Tensor,
                len_a: torch.Tensor,
                seq_b: torch.Tensor,
                len_b: torch.Tensor,
                endpoint_features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Siamese network

        Args:
            seq_a: Padded sequences A (batch, max_len_a, input_size)
            len_a: Lengths of sequences A (batch,)
            seq_b: Padded sequences B (batch, max_len_b, input_size)
            len_b: Lengths of sequences B (batch,)
            endpoint_features: Optional endpoint features (batch, endpoint_feature_dim)
                             Contains: [time_gap, x_gap, y_gap, velocity_diff]

        Returns:
            similarity: Probability of same vehicle (batch, 1)
            emb_a: Embedding of fragment A (batch, embedding_size)
            emb_b: Embedding of fragment B (batch, embedding_size)
        """
        # Encode both fragments using shared encoder
        emb_a = self.encoder(seq_a, len_a)
        emb_b = self.encoder(seq_b, len_b)

        # 2. Symmetric feature combination (Ensures S(A,B) = S(B,A))
        diff = torch.abs(emb_a - emb_b)
        prod = emb_a * emb_b
        
        # Concatenate symmetric features with optional endpoint features
        if self.use_endpoint_features and endpoint_features is not None:
            combined = torch.cat([diff, prod, endpoint_features], dim=1)
        else:
            combined = torch.cat([diff, prod], dim=1)

        # Compute similarity score
        similarity = self.similarity_head(combined)

        return similarity, emb_a, emb_b


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks

    Pulls together embeddings of similar pairs (label=1)
    Pushes apart embeddings of dissimilar pairs (label=0)
    """

    def __init__(self, margin: float = 2.0):
        """
        Args:
            margin: Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            emb_a: Embeddings of fragment A (batch, embedding_size)
            emb_b: Embeddings of fragment B (batch, embedding_size)
            labels: 1 for same vehicle, 0 for different (batch, 1)

        Returns:
            loss: Scalar loss value
        """
        # Euclidean distance between embeddings
        distances = F.pairwise_distance(emb_a, emb_b)

        # Contrastive loss formula
        # For similar pairs (label=1): penalize large distances
        # For dissimilar pairs (label=0): penalize small distances (below margin)
        labels = labels.squeeze()
        loss_similar = labels * torch.pow(distances, 2)
        loss_dissimilar = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        loss = torch.mean(loss_similar + loss_dissimilar)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Binary Cross-Entropy + Contrastive Loss

    BCE loss trains the similarity head directly (Primary for inference)
    Contrastive loss shapes the embedding space (Secondary regularization)
    """

    def __init__(self, margin: float = 2.0, alpha: float = 0.1):
        """
        Args:
            margin: Margin for contrastive loss
            alpha: Weight for contrastive loss (1-alpha for BCE). 
                   Set alpha=0 to use BCE only.
        """
        super(CombinedLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha

    def forward(self,
                similarity: torch.Tensor,
                emb_a: torch.Tensor,
                emb_b: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss

        Args:
            similarity: Predicted similarity (batch, 1)
            emb_a: Embeddings of fragment A
            emb_b: Embeddings of fragment B
            labels: Ground truth (batch, 1)

        Returns:
            total_loss: Combined loss
            bce: BCE component
            contrastive: Contrastive component
        """
        bce = self.bce_loss(similarity, labels)
        contrastive = self.contrastive_loss(emb_a, emb_b, labels)

        total_loss = (1 - self.alpha) * bce + self.alpha * contrastive

        return total_loss, bce, contrastive


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model architecture
    print("Testing Siamese Network Architecture...")

    # Create model
    model = SiameseTrajectoryNetwork(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        similarity_hidden_dim=64
    )

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass with dummy data
    batch_size = 8
    seq_len_a = 20
    seq_len_b = 15
    input_size = 4

    # Create dummy sequences
    seq_a = torch.randn(batch_size, seq_len_a, input_size)
    len_a = torch.LongTensor([seq_len_a] * batch_size)
    seq_b = torch.randn(batch_size, seq_len_b, input_size)
    len_b = torch.LongTensor([seq_len_b] * batch_size)
    endpoint_features = torch.randn(batch_size, 4)
    labels = torch.randint(0, 2, (batch_size, 1)).float()

    # Forward pass
    similarity, emb_a, emb_b = model(seq_a, len_a, seq_b, len_b, endpoint_features)

    print(f"\nForward pass test:")
    print(f"  Input A shape: {seq_a.shape}")
    print(f"  Input B shape: {seq_b.shape}")
    print(f"  Embedding A shape: {emb_a.shape}")
    print(f"  Embedding B shape: {emb_b.shape}")
    print(f"  Similarity shape: {similarity.shape}")
    print(f"  Similarity values: {similarity.squeeze()}")

    # Test loss
    criterion = CombinedLoss(margin=2.0, alpha=0.5)
    total_loss, bce, contrastive = criterion(similarity, emb_a, emb_b, labels)

    print(f"\nLoss test:")
    print(f"  BCE Loss: {bce.item():.4f}")
    print(f"  Contrastive Loss: {contrastive.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")

    print("\n✓ Model architecture test passed!")
