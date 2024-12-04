import torch
import torch.nn as nn
import math

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, cond_drop_rate=0.1):
        """
        Class embedding layer for Classifier-Free Guidance (CFG).

        Args:
            embed_dim (int): Dimension of the embedding vector.
            n_classes (int): Number of classes.
            cond_drop_rate (float): Conditional dropout rate for CFG.
        """
        super().__init__()
        
        # Embedding layer for class embeddings
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = n_classes

    def forward(self, x):
        """
        Forward pass for class embeddings.

        Args:
            x (torch.Tensor): Tensor of class indices of shape (batch_size,).

        Returns:
            c (torch.Tensor): Class embeddings of shape (batch_size, embed_dim).
        """
        b = x.shape[0]
        
        if self.cond_drop_rate > 0 and self.training:
            # Implement class drop for unconditional class
            mask = torch.rand(b, device=x.device) < self.cond_drop_rate
            x = x.clone()  # Avoid modifying the input tensor directly
            x[mask] = self.num_classes - 1  # Use a reserved "unconditional" class index

        # Get the embedding for the input class indices
        c = self.embedding(x)
        return c
