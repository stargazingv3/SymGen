#!/usr/bin/env python3
"""
Batch prediction wrapper around predict.py that preserves function identifiers.

This script loads the model once and processes all functions, preserving
the function_id for later mapping back to original functions.
"""

import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, CodeLlamaTokenizer

from utils.prompter import Prompter

import json


def check_gpu_availability():
    """Check GPU availability and print diagnostics."""
    if not torch.cuda.is_available():
        print("[!] CUDA is NOT available - will use CPU")
        print(f"[!] PyTorch version: {torch.__version__}")
        print(f"[!] CUDA compiled version: {torch.version.cuda}")
        return "cpu"
    
    device = "cuda"
    gpu_count = torch.cuda.device_count()
    print(f"[+] CUDA is available!")
    print(f"[+] Number of GPUs: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"[+] GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    return device


device = check_gpu_availability()


def main(
    input_path: str = "",
    output_dir: str = "",
    base_model: str = "",
    lora_weights: str = "",
    load_8bit: bool = False,
    prompt_template: str = "codellama",
):
    """Run batch predictions while preserving function identifiers."""
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        input_path
    ), "Please specify a --input_path"
    assert (
        output_dir
    ), "Please specify a --output_dir"
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='codellama/CodeLlama-34b-Instruct-hf'"
    assert (
        lora_weights
    ), "Please specify a --lora_weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    prompter = Prompter(prompt_template)
    tokenizer = CodeLlamaTokenizer.from_pretrained(base_model)
    
    if device != "cuda":
        print("[!] WARNING: No CUDA device available. This model is too large for CPU inference.")
        print("[!] Attempting to load anyway, but this will be extremely slow...")
        model = LlamaForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/app/.cache/huggingface"),
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    else:
        # Use automatic device mapping for multi-GPU support
        print(f"[+] Loading model with device_map='auto' for multi-GPU support...")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",  # Automatically distribute across all available GPUs
            cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/app/.cache/huggingface"),
        )
        print(f"[+] Loading LoRA weights...")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    # Print device mapping diagnostics
    if hasattr(model, 'hf_device_map'):
        print(f"[+] Model device_map: {model.hf_device_map}")
        if len(set(model.hf_device_map.values())) > 1:
            print(f"[+] Model using multi-GPU device_map across {len(set(model.hf_device_map.values()))} devices")
        else:
            print(f"[+] Model using single device: {list(model.hf_device_map.values())[0]}")
    else:
        device_map = getattr(model, 'device_map', None)
        print(f"[+] Model device_map: {device_map}")
        if device_map:
            print(f"[+] Model using explicit device_map")
    
    # Check which device the model parameters are on
    try:
        sample_param = next(model.parameters())
        print(f"[+] Sample parameter device: {sample_param.device}")
    except StopIteration:
        print("[!] Could not determine parameter device")
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit and device != "cuda":
        model.half()  # Only apply half precision if not using CUDA auto device_map

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32" and device == "cpu":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        stream_output=False,
        **kwargs,
    ):
        """Evaluate a single function and return predicted name."""
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # When using device_map="auto", move to first available device
        # or use the device where the first layer resides
        if device == "cuda" and hasattr(model, 'hf_device_map'):
            # Multi-GPU: use the device of the first model layer
            first_device = list(model.hf_device_map.values())[0] if model.hf_device_map else device
            input_ids = inputs["input_ids"].to(first_device)
            attention_mask = inputs['attention_mask'].to(first_device)
            print(f"[+] Input device: {first_device}", end='\r')
        else:
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        full_response = prompter.get_response(output)
        
        # Extract just the function name (clean up common prefixes/suffixes)
        predicted_name = full_response.strip()
        # Remove common prefixes like "The predicted function name is "
        if "predicted function name is" in predicted_name.lower():
            predicted_name = predicted_name.split("is", 1)[-1].strip()
        # Remove trailing </s> or other tokens
        predicted_name = predicted_name.split("</s>")[0].strip()
        predicted_name = predicted_name.split("\n")[0].strip()
        
        return predicted_name, full_response

    # Load input data
    with open(input_path) as f:
        testset = json.load(f)

    predictions = []
    total = len(testset)
    
    print(f'[+] Processing {total} functions...')
    
    for idx, t in enumerate(testset):
        function_id = t.get("function_id", f"function_{idx}")
        # Use instruction from testset (should be set by convert_to_prediction_format.py)
        instruction = t.get("instruction", "Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, you should infer code semantics and tell me the original function name from the contents of the function to replace [MASK]. And you need to tell me your answer. Now the decompiled codes are as follows:")
        input_code = t.get("input", "")
        
        if idx % 10 == 0:
            print(f'[+] Processing function {idx + 1}/{total}: {function_id}')
        
        # Evaluate using instruction and input from testset (matching predict.py behavior)
        predicted_name, full_response = evaluate(instruction, input_code)
        
        prediction_entry = {
            "function_id": function_id,
            "predicted_name": predicted_name,
            "full_response": full_response,
            "source_file": t.get("source_file", "")
        }
        predictions.append(prediction_entry)

    # Save predictions
    output_file = os.path.join(output_dir, 'predictions.json')
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f'[+] Saved {len(predictions)} predictions to {output_file}')


if __name__ == "__main__":
    fire.Fire(main)

