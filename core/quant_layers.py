import torch
import bitsandbytes as bnb

class CircuitAwareLinear(bnb.nn.Linear4bit):
    """
    A mixed-precision linear layer that processes a subset of neurons in FP16
    and the remaining neurons in 4-bit quantized precision.
    """
    def __init__(self, importance_scores: torch.Tensor, fp16_ratio: float = 0.2,
                 in_features: int = None, out_features: int = None, bias: bool = True):
        if in_features is None or out_features is None:
            raise ValueError("Both in_features and out_features must be specified.")
        # Initialize the 4-bit linear layer from bitsandbytes.
        super().__init__(in_features, out_features, bias=bias)
        self.importance = importance_scores
        self.fp16_indices = self._get_fp16_indices(fp16_ratio)
        
    def _get_fp16_indices(self, ratio: float):
        """Select top-k neurons based on the importance scores to keep in FP16."""
        k = int(self.in_features * ratio)
        topk = torch.topk(self.importance, k=k)
        # Create a boolean mask: True where the neuron is in the top-k
        mask = torch.zeros(self.in_features, dtype=torch.bool, device=self.importance.device)
        mask[topk.indices] = True
        return mask
        
    def forward(self, x: torch.Tensor):
        """
        Split the input into two branches based on fp16_indices.
        Processes the important neurons in FP16 and the rest using quantized matmul.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError("Input tensor feature dimension does not match in_features.")
        fp16_mask = self.fp16_indices
        quant_mask = ~self.fp16_indices
        
        # Split input accordingly
        x_fp16 = x[:, fp16_mask]
        x_quant = x[:, quant_mask]
        
        # Process the FP16 branch using high-precision weights.
        w_fp16 = self.weight[fp16_mask].to(x_fp16.dtype)
        out_fp16 = torch.matmul(x_fp16, w_fp16.T)
        
        # Process the quantized branch using the parent's forward.
        # Backup original weights then substitute with subset corresponding to quant_mask.
        original_weight = self.weight
        self.weight = self.weight[quant_mask]
        out_quant = super().forward(x_quant)
        self.weight = original_weight
        
        return out_fp16 + out_quant

class AdaptiveCircuitQuant(CircuitAwareLinear):
    """
    An extension of CircuitAwareLinear that dynamically updates importance scores during fine-tuning.
    (Note: Overriding backward is non-standard in PyTorch. In practice, you would use 
    hooks to update the scores. This is a pedagogical example.)
    """
    def backward(self, grad_output):
        # Update importance scores based on the gradient signal.
        grad_update = grad_output.abs().mean(dim=0)
        self.importance = self.importance + grad_update
        return super().backward(grad_output)