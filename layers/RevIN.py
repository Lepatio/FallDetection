import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
    """
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        """
        Initialize RevIN
        
        Args:
            num_features: Number of features/channels in the input
            eps: A small constant for numerical stability
            affine: If True, RevIN has learnable affine parameters
            subtract_last: If True, subtract the last value instead of the mean
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
        self.mean = None
        self.stdev = None
            
    def forward(self, x, mode='norm'):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels]
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Transformed tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Mode {mode} not recognized. Must be 'norm' or 'denorm'")
    
    def _normalize(self, x):
        """
        Normalize input
        """
        # Mean along the sequence length dimension
        if self.subtract_last:
            # Use the last value for normalization
            self.last = x[:, -1:, :].detach()
            x = x - self.last
        else:
            # Use mean for normalization
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - self.mean
        
        # Standard deviation along the sequence length dimension
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = x / self.stdev
        
        # Apply affine transformation if needed
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
            
        return x
    
    def _denormalize(self, x):
        """
        Denormalize input (reverse the normalization)
        """
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
            
        x = x * self.stdev
        
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
            
        return x