import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM
from captum.attr import LayerIntegratedGradients

class LlamaImportanceScorer:
    def __init__(self, model: LlamaForCausalLM):
        self.model = model.to("cuda")  # Ensure model is on GPU
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_layer_attributions(self, layer_idx: int, inputs: torch.Tensor):
        """Compute per-layer attributions using LayerIntegratedGradients."""
        layer = self.model.model.layers[layer_idx]
        lig = LayerIntegratedGradients(self.model, layer)

        baseline = torch.zeros_like(inputs).to(inputs.device)  # Ensure baseline is on the same device
        attributions = lig.attribute(
            inputs=inputs,
            baselines=baseline,
            n_steps=20,
            return_convergence_delta=False
        )
        return attributions.mean(dim=0)

    def score_all_layers(self, calib_data: torch.Tensor):
        """Compute importance scores for all layers given calibration data."""
        scores = {}
        num_layers = self.model.config.num_hidden_layers
        for idx in tqdm(range(num_layers), desc="Scoring Layers"):
            score = self._get_layer_attributions(idx, calib_data)
            scores[f"layer_{idx}"] = score.detach().cpu()  # Move scores to CPU for later use
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
