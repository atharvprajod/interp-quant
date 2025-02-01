import torch

def dynamic_head_selection(query: torch.Tensor, k: int = 4):
    """
    Select the top-k attention heads per token dynamically based on a scoring function.
    
    Args:
        query: Tensor of shape (batch, heads, seq_len, d_model).
        k: Number of heads to select.
    
    Returns:
        top_heads: Indices corresponding to the top attention heads.
    """
    # Compute a score for each head. Here we use a simple dot-product-based approach.
    # (score shape: [batch, heads, seq_len])
    scores = torch.einsum("bhnd,bhnd->bhn", query, query.mean(dim=-1, keepdim=True).expand_as(query))
    # Select top-k heads along the heads dimension.
    top_heads = torch.topk(scores, k=k, dim=1).indices
    return top_heads

def load_quant_config(config_file: str):
    """
    Load a YAML configuration file containing quantization thresholds and parameters.
    
    Returns:
        A dictionary with configuration values.
    """
    import yaml
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config