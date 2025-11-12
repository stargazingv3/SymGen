#!/usr/bin/env python3
"""
Convert processed decompiled data (from process_decompiled_data.py -p) 
into the JSON array format required by predict.py.

Input: JSON files with {function_name: masked_code_string}
Output: JSON array with [{instruction, input, output, function_id}]
"""

import os
import json
import argparse
import sys

# Standard instruction template from predict.py line 154
INSTRUCTION_TEMPLATE = "Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, you should infer code semantics and tell me the original function name from the contents of the function to replace [MASK]. And you need to tell me your answer. Now the decompiled codes are as follows:"


def convert_file(input_file, output_file):
    """Convert a single processed JSON file to prediction format."""
    print(f'[+] Processing {input_file}')
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    prediction_data = []
    
    for function_id, masked_code in data.items():
        entry = {
            "instruction": INSTRUCTION_TEMPLATE,
            "input": masked_code,
            "output": "",
            "function_id": function_id
        }
        prediction_data.append(entry)
    
    with open(output_file, 'w') as f:
        json.dump(prediction_data, f, indent=4)
    
    print(f'[+] Wrote {len(prediction_data)} functions to {output_file}')
    return len(prediction_data)


def main(input_path, output_dir, single_file=False):
    """Convert processed JSON files to prediction format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_path):
        # Single file
        filename = os.path.basename(input_path)
        output_file = os.path.join(output_dir, filename)
        convert_file(input_path, output_file)
    elif os.path.isdir(input_path):
        # Directory - walk through all JSON files
        all_predictions = []
        total_functions = 0
        
        for root, dirs, files in os.walk(input_path):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                
                input_file = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_path)
                
                if single_file:
                    # Collect all for single output file
                    with open(input_file, 'r') as f:
                        data = json.load(f)
                    
                    for function_id, masked_code in data.items():
                        entry = {
                            "instruction": INSTRUCTION_TEMPLATE,
                            "input": masked_code,
                            "output": "",
                            "function_id": function_id,
                            "source_file": os.path.join(relative_path, filename) if relative_path != '.' else filename
                        }
                        all_predictions.append(entry)
                        total_functions += 1
                else:
                    # Per-file output
                    if relative_path == '.':
                        output_file = os.path.join(output_dir, filename)
                    else:
                        new_output_dir = os.path.join(output_dir, relative_path)
                        if not os.path.exists(new_output_dir):
                            os.makedirs(new_output_dir, exist_ok=True)
                        output_file = os.path.join(new_output_dir, filename)
                    
                    total_functions += convert_file(input_file, output_file)
        
        if single_file:
            output_file = os.path.join(output_dir, 'all_predictions.json')
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=4)
            print(f'[+] Wrote {total_functions} functions to {output_file}')
    else:
        print(f'[!] Error: {input_path} is not a valid file or directory')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert processed decompiled data to prediction format'
    )
    parser.add_argument(
        '--input_path', '-i',
        type=str,
        required=True,
        help='Path to processed JSON file or directory (output from process_decompiled_data.py -p)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Directory to save prediction format JSON files'
    )
    parser.add_argument(
        '--single_file', '-s',
        action='store_true',
        help='Combine all functions into a single output file'
    )
    
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.single_file)

