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
    input_path: str = "",
    output_dir: str = "",
    base_model: str = "",
    lora_weights: str = "",
    load_8bit: bool = False,
    prompt_template: str = "codellama",  # The prompt template to use, will default to alpaca.
):
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
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
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
        num_beams=1,
        max_new_tokens=256,
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

    
    with open(os.path.join(output_dir, 'predicted_function_name.json'), 'w') as f:
        json.dump([], f, indent=4)

    with open(input_path) as f:
        testset = json.load(f)

    it = 0
    for t in testset:
        instruction = "Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, you should infer code semantics and tell me the original function name from the contents of the function to replace [MASK]. And you need to tell me your answer. Now the decompiled codes are as follows:"
        predicted_name = evaluate(t["instruction"], t["input"])
        print('-' * 20, "Test Case", it, '-' * 20,)
        it = it + 1
        print(t["instruction"], t["input"])
        print(predicted_name)
        print()

        new_data = {}
        new_data['ground_truth'] = t["output"]
        new_data['predicted_name'] = predicted_name

        # with open(os.path.join(output_dir, 'predicted_function_name.json'), 'w') as f:
        #     json.dump(inference_result, f, indent=4)

        with open(os.path.join(output_dir, 'predicted_function_name.json'), 'r+') as f:
            data_for_update = json.load(f)
            data_for_update.append(new_data)
            f.seek(0)
            f.truncate()
            json.dump(data_for_update, f, indent=4)
        

if __name__ == "__main__":
    fire.Fire(main)
