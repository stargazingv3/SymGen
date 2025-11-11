import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, CodeLlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import json


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    input_file: str = "",
    output_dir: str = "",
    base_model: str = "",
    load_8bit: bool = False,
    prompt_template: str = "codellama",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='codellama/CodeLlama-34b-Instruct-hf'"
    assert (
        input_file
    ), "Please specify a --input_file, e.g. --input_file='dataset/test_set.json'"
    assert (
        output_dir
    ), "Please specify a --output_dir"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    prompter = Prompter(prompt_template)
    tokenizer = CodeLlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/app/.cache/huggingface"),
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
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
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
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

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

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


    with open(input_file, 'r') as f:
        source_functions = json.load(f)

    with open(os.path.join(output_dir, 'function_summary.json'), 'w') as f:
        json.dump([], f, indent=4)

    for function_name in source_functions.keys():
        instruction = "Summarize the function provided below in a concise and clear manner in 512 words. Highlight the key inputs, outputs, main steps and the main purpose of the function. Avoid unnecessary details and focus on delivering a high-level overview."
        res = evaluate(instruction, source_functions[function_name])
        print('-' * 20, 'Function Summary:', function_name, '-' * 20,)
        print(res)

        new_data = {}
        new_data['instruction'] = instruction
        new_data['input'] = source_functions[function_name]
        new_data['output'] = res

        with open(os.path.join(output_dir, 'function_summary.json'), 'r+') as f:
            data_for_update = json.load(f)
            data_for_update.append(new_data)
            f.seek(0)
            f.truncate()
            json.dump(data_for_update, f, indent=4)
        

if __name__ == "__main__":
    fire.Fire(main)
