#!/usr/bin/env python3
"""
Generate function summaries for all functions in mapped predictions JSON.

Reuses model loading and evaluation logic from generate_summary.py.
Adds summary field to each function in the mapped predictions JSON.
"""

import os
import sys

import fire
import torch
import transformers
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
    input_file: str = "",
    output_file: str = "",
    base_model: str = "",
    load_8bit: bool = False,
    prompt_template: str = "codellama",
):
    """Generate summaries for all functions in mapped predictions JSON."""
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='codellama/CodeLlama-34b-Instruct-hf'"
    assert (
        input_file
    ), "Please specify a --input_file (mapped predictions JSON)"
    assert (
        output_file
    ), "Please specify a --output_file"

    # Load mapped functions
    print(f'[+] Loading mapped functions from {input_file}')
    with open(input_file, 'r') as f:
        mapped_functions = json.load(f)

    # Load model and tokenizer
    print(f'[+] Loading model {base_model}')
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
    
    # Print device mapping diagnostics
    if hasattr(model, 'hf_device_map'):
        print(f"[+] Model device_map: {model.hf_device_map}")
        if len(set(model.hf_device_map.values())) > 1:
            print(f"[+] Model using multi-GPU device_map across {len(set(model.hf_device_map.values()))} devices")
        else:
            print(f"[+] Model using single device: {list(model.hf_device_map.values())[0]}")
    
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
        num_beams=3,
        max_new_tokens=512,
        stream_output=False,
        **kwargs,
    ):
        """Evaluate and return summary."""
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # When using device_map="auto", move to first available device
        if device == "cuda" and hasattr(model, 'hf_device_map'):
            # Multi-GPU: use the device of the first model layer
            first_device = list(model.hf_device_map.values())[0] if model.hf_device_map else device
            input_ids = inputs["input_ids"].to(first_device)
            attention_mask = inputs['attention_mask'].to(first_device)
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
        return prompter.get_response(output)

    # Summary instruction template from generate_summary.py line 137
    summary_instruction = "Summarize the function provided below in a concise and clear manner in 512 words. Highlight the key inputs, outputs, main steps and the main purpose of the function. Avoid unnecessary details and focus on delivering a high-level overview."

    total = len(mapped_functions)
    processed = 0
    
    print(f'[+] Generating summaries for {total} functions...')
    
    # Process each function
    for addr, func_data in mapped_functions.items():
        processed += 1
        function_name = func_data.get("predicted_name") or func_data.get("original_name", addr)
        
        if processed % 10 == 0:
            print(f'[+] Processing {processed}/{total}: {function_name} @ {addr}')
        
        # Get decompiled code
        decomp_code = func_data.get("decomp_code", "")
        if not decomp_code:
            print(f'[!] Warning: No decompiled code for {function_name} @ {addr}, skipping')
            func_data["summary"] = ""
            continue
        
        # Generate summary
        try:
            summary = evaluate(summary_instruction, decomp_code)
            func_data["summary"] = summary
        except Exception as e:
            print(f'[!] Error generating summary for {function_name} @ {addr}: {e}')
            func_data["summary"] = ""
    
    # Save updated mapped functions with summaries
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(mapped_functions, f, indent=4)
    
    print(f'[+] Saved {processed} functions with summaries to {output_file}')


if __name__ == "__main__":
    fire.Fire(main)

