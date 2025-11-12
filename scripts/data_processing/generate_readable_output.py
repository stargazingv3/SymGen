#!/usr/bin/env python3
"""
Generate human-readable text file similar to Ghidra/IDA decompilation view.

Creates a single searchable text file with:
- Predicted function names
- Function summaries
- Assembly code with addresses
- Decompiled code with predicted names
"""

import os
import json
import argparse
import sys
import re


def parse_address(address_str):
    """Parse address string to integer for sorting."""
    try:
        if address_str.startswith('0x'):
            return int(address_str, 16)
        elif address_str.startswith('0'):
            return int(address_str, 16)
        else:
            return int(address_str, 16)
    except:
        return 0


def replace_function_name_in_code(decomp_code, predicted_name, original_name):
    """Replace [MASK] or original function name with predicted name in decompiled code."""
    if not predicted_name:
        return decomp_code
    
    # Replace [MASK] with predicted name
    decomp_code = decomp_code.replace('[MASK]', predicted_name)
    
    # Replace original function name if it exists
    if original_name and original_name in decomp_code:
        # Use regex to replace function name in function definition
        # Pattern: return_type original_name(parameters)
        pattern = r'\b' + re.escape(original_name) + r'\s*\('
        replacement = predicted_name + '('
        decomp_code = re.sub(pattern, replacement, decomp_code)
    
    return decomp_code


def format_assembly(assembly_list):
    """Format assembly list into readable text."""
    if not assembly_list:
        return "No assembly available"
    
    lines = []
    for asm_line in assembly_list:
        lines.append(f"  {asm_line}")
    
    return "\n".join(lines)


def generate_readable_output(mapped_functions_json, output_file, binary_name=None):
    """Generate human-readable output file."""
    print(f'[+] Loading mapped functions from {mapped_functions_json}')
    with open(mapped_functions_json, 'r') as f:
        mapped_functions = json.load(f)
    
    # Sort functions by address
    sorted_addresses = sorted(mapped_functions.keys(), key=parse_address)
    
    print(f'[+] Generating readable output for {len(sorted_addresses)} functions...')
    
    output_lines = []
    
    # Header
    if binary_name:
        output_lines.append(f"Binary Analysis: {binary_name}")
        output_lines.append("=" * 80)
        output_lines.append("")
    
    # Process each function
    for addr in sorted_addresses:
        func_data = mapped_functions[addr]
        
        original_name = func_data.get("original_name", "")
        predicted_name = func_data.get("predicted_name", "")
        summary = func_data.get("summary", "")
        decomp_code = func_data.get("decomp_code", "")
        assembly = func_data.get("assembly", [])
        func_addr = func_data.get("function_address", {})
        start_addr = func_addr.get("start", addr)
        end_addr = func_addr.get("end", "")
        
        # Function header
        output_lines.append("=" * 80)
        if predicted_name:
            display_name = predicted_name
            if original_name and original_name != predicted_name:
                display_name += f" (predicted from {original_name})"
        else:
            display_name = original_name or addr
        
        output_lines.append(f"Function: {display_name}")
        if end_addr:
            output_lines.append(f"Address: {start_addr} - {end_addr}")
        else:
            output_lines.append(f"Address: {start_addr}")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        # Summary
        if summary:
            output_lines.append("Summary:")
            output_lines.append(summary)
            output_lines.append("")
        
        # Assembly
        if assembly:
            output_lines.append("Assembly:")
            output_lines.append(format_assembly(assembly))
            output_lines.append("")
        
        # Decompiled Code
        if decomp_code:
            output_lines.append("Decompiled Code:")
            # Replace function name in decompiled code
            updated_code = replace_function_name_in_code(
                decomp_code, predicted_name, original_name
            )
            output_lines.append(updated_code)
            output_lines.append("")
        
        output_lines.append("")
    
    # Write output file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f'[+] Saved readable output to {output_file}')
    print(f'[+] Total functions: {len(sorted_addresses)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate human-readable output file from mapped predictions'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to mapped functions JSON (with summaries)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to save readable output text file'
    )
    parser.add_argument(
        '--binary_name', '-b',
        type=str,
        default=None,
        help='Optional binary name to include in header'
    )
    
    args = parser.parse_args()
    generate_readable_output(args.input, args.output, args.binary_name)

