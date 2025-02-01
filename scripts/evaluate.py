#!/usr/bin/env python
import torch
import time
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer

def evaluate_model(model, tokenizer, task_texts):
    """
    Evaluate model accuracy and speed for a list of input texts.
    Returns:
        outputs: Generated texts.
        tokens_per_sec: Speed metric.
    """
    model.eval()
    total_time = 0
    total_tokens = 0
    outputs = []
    
    for text in task_texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        start = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_length=input_ids.shape[-1] + 20)
        end = time.time()
        total_time += (end - start)
        total_tokens += output.shape[-1]
        outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    return outputs, tokens_per_sec

def main():
    parser = argparse.ArgumentParser(description="Evaluate Quantized LLaMA Model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the quantized model directory")
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "hellaswag"],
                        help="List of task names for evaluation")
    args = parser.parse_args()

    print("Loading model from", args.model)
    model = LlamaForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Dummy task examples
    task_texts = {
        "mmlu": "What is the capital of France?",
        "hellaswag": "A man is eating a meal. The man is shown to be eating a meal by the camera because"
    }
    
    for task in args.tasks:
        if task in task_texts:
            print(f"\nEvaluating task: {task}")
            texts = [task_texts[task]] * 10  # Evaluate on 10 samples
            outputs, tokens_per_sec = evaluate_model(model, tokenizer, texts)
            print(f"Task {task}: {tokens_per_sec:.2f} tokens/sec")
            print("Sample output:", outputs[0])
        else:
            print(f"Task {task} is not defined. Skipping.")

if __name__ == "__main__":
    main()