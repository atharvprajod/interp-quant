import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM
from captum.attr import LayerIntegratedGradients

class LlamaImportanceScorer:
    def __init__(self, model: LlamaForCausalLM):
        self.model = model
        self.model.eval()  # Ensure evaluation mode for attribution
        for param in self.model.parameters():
            param.requires_grad = False  # Disable gradients

    def _get_layer_attributions(self, layer_idx: int, inputs: torch.Tensor):
        """Compute per-layer attributions using LayerIntegratedGradients."""
        # Fetch the appropriate transformer layer
        layer = self.model.model.layers[layer_idx]
        lig = LayerIntegratedGradients(self.model, layer)
        
        # Create a baseline (all zeros) for attribution calculation
        baseline = torch.zeros_like(inputs)
        attributions = lig.attribute(
            inputs=inputs,
            baselines=baseline,
            n_steps=20,
            return_convergence_delta=False
        )
        # Aggregate across batch and sequence dimensions (resulting in a d_model vector)
        return attributions.mean(dim=0)

    def score_all_layers(self, calib_data: torch.Tensor):
        """Compute importance scores for all layers given calibration data."""
        scores = {}
        num_layers = self.model.config.num_hidden_layers
        for idx in tqdm(range(num_layers), desc="Scoring Layers"):
            score = self._get_layer_attributions(idx, calib_data)
            # Detach and move to CPU so that later modules can use the scores
            scores[f"layer_{idx}"] = score.detach().cpu()
        return scores

def trace_attention_paths(model, input_ids, num_heads=4):
    """
    Identify critical attention heads via causal mediation.
    Args:
        model: The transformer model.
        input_ids: Tensor of input token IDs.
        num_heads: (Optional) Number of top heads to trace.
    Returns:
        head_importance: A tensor where higher values indicate more critical heads.
    """
    model.eval()
    original_output = model(input_ids).logits
    total_heads = model.config.num_attention_heads
    head_importance = torch.zeros(total_heads)
    
    for head in range(total_heads):
        # Assume the model accepts an ablation parameter for individual heads.
        with torch.no_grad():
            modified_output = model(input_ids, ablate_attention_head=head).logits
            delta = (original_output - modified_output).abs().sum().item()
            head_importance[head] = delta
            
    return head_importance
