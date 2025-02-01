#!/usr/bin/env python
import torch
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
from core.importance_scorer import LlamaImportanceScorer

def main():
    parser = argparse.ArgumentParser(
        description="Calibration script for LLaMA-3.8B quantization.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Pretrained model identifier or path")
    parser.add_argument("--output", type=str, default="scores/llama3-8b_importance_scores.pt",
                        help="Path to save computed importance scores")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # Prepare calibration data
    print("Preparing calibration data...")
    texts = ["This is a sample calibration input for LLaMA quantization."] * 32
    calib_data = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = calib_data.input_ids.to(model.device)

    # Compute importance scores layer-by-layer
    scorer = LlamaImportanceScorer(model)
    print("Scoring layers...")
    layer_scores = scorer.score_all_layers(input_ids)
    
    print(f"Saving importance scores to {args.output}")
    torch.save(layer_scores, args.output)
    print("Calibration complete.")

if __name__ == "__main__":
    main()