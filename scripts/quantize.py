#!/usr/bin/env python
import torch
import argparse
from transformers import LlamaForCausalLM
from interp_quant.core.quant_layers import CircuitAwareLinear

def apply_circuit_quant(model, layer_scores, fp16_ratio=0.2):
    """
    Replace key submodules in each transformer layer with their circuit-aware
    quantized versions.
    """
    for idx, layer in enumerate(model.model.layers):
        score_key = f"layer_{idx}"
        if score_key in layer_scores:
            # Replace the MLP gate projection if it exists.
            if hasattr(layer.mlp, "gate_proj"):
                in_features = layer.mlp.gate_proj.in_features
                out_features = layer.mlp.gate_proj.out_features
                importance_score = layer_scores[score_key].to(layer.mlp.gate_proj.weight.device)
                layer.mlp.gate_proj = CircuitAwareLinear(
                    importance_scores=importance_score,
                    fp16_ratio=fp16_ratio,
                    in_features=in_features,
                    out_features=out_features,
                    bias=True
                )
            # Replace the self-attention query projection if it exists.
            if hasattr(layer.self_attn, "q_proj"):
                in_features = layer.self_attn.q_proj.in_features
                out_features = layer.self_attn.q_proj.out_features
                importance_score = layer_scores[score_key].to(layer.self_attn.q_proj.weight.device)
                layer.self_attn.q_proj = CircuitAwareLinear(
                    importance_scores=importance_score,
                    fp16_ratio=fp16_ratio,
                    in_features=in_features,
                    out_features=out_features,
                    bias=True
                )
    return model

def main():
    parser = argparse.ArgumentParser(description="Apply Circuit-Aware Quantization")
    parser.add_argument("--checkpoint", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Pretrained model checkpoint")
    parser.add_argument("--scores", type=str, default="scores/llama3-8b_importance_scores.pt",
                        help="Path to precomputed importance scores")
    parser.add_argument("--output", type=str, default="quantized-llama3",
                        help="Directory to save the quantized model")
    parser.add_argument("--fp16_ratio", type=float, default=0.2,
                        help="Fraction of neurons to process in FP16")
    args = parser.parse_args()

    print("Loading model from", args.checkpoint)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint)
    print("Loading importance scores from", args.scores)
    layer_scores = torch.load(args.scores)

    print("Applying circuit-aware quantization...")
    model = apply_circuit_quant(model, layer_scores, args.fp16_ratio)
    
    print("Saving quantized model to", args.output)
    # Save the modified model
    model.save_pretrained(args.output)
    print("Quantization complete.")

if __name__ == "__main__":
    main()