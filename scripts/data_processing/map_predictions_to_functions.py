#!/usr/bin/env python3
"""
Map predictions back to original Ghidra decompilation data.

Combines predictions JSON with original Ghidra JSON to create a comprehensive
analysis result with all metadata, assembly, and decompiled code.
"""

import os
import json
import argparse
import sys


def parse_address(address_str):
    """Parse address string to integer for sorting."""
    try:
        # Handle formats like "0x401234" or "00401234"
        if address_str.startswith('0x'):
            return int(address_str, 16)
        elif address_str.startswith('0'):
            return int(address_str, 16)
        else:
            return int(address_str, 16)
    except:
        return 0


def match_function(prediction, ghidra_data):
    """
    Match a prediction to a function in Ghidra data.
    Returns the function address (key) if found, None otherwise.
    """
    function_id = prediction.get("function_id", "")
    
    # Try matching by address first (most common case - function_id is typically an address)
    if function_id in ghidra_data:
        return function_id
    
    # Try direct match by function name/ID
    for addr, func_data in ghidra_data.items():
        if func_data.get("func_name") == function_id:
            return addr
    
    # Try partial match (e.g., FUN_00001234 might match address 0x1234)
    if function_id.startswith("FUN_"):
        try:
            # Extract hex part from FUN_00001234
            hex_part = function_id.replace("FUN_", "").replace("0x", "")
            # Try to find matching address
            for addr in ghidra_data.keys():
                addr_hex = addr.replace("0x", "").replace("0", "").lstrip("0")
                if addr_hex.endswith(hex_part) or hex_part in addr:
                    return addr
        except:
            pass
    
    return None


def main(ghidra_json_path, predictions_json_path, output_path):
    """Map predictions to original Ghidra functions."""
    print(f'[+] Loading Ghidra data from {ghidra_json_path}')
    with open(ghidra_json_path, 'r') as f:
        ghidra_data = json.load(f)
    
    print(f'[+] Loading predictions from {predictions_json_path}')
    with open(predictions_json_path, 'r') as f:
        predictions = json.load(f)
    
    # Create mapping from function_id to prediction
    prediction_map = {}
    for pred in predictions:
        function_id = pred.get("function_id", "")
        prediction_map[function_id] = pred
    
    # Map predictions to functions
    mapped_functions = {}
    matched_count = 0
    unmatched_predictions = []
    
    for pred in predictions:
        function_id = pred.get("function_id", "")
        matched_addr = match_function(pred, ghidra_data)
        
        if matched_addr:
            matched_count += 1
            func_data = ghidra_data[matched_addr].copy()
            func_data["original_name"] = func_data.get("func_name", function_id)
            func_data["predicted_name"] = pred.get("predicted_name", "")
            func_data["prediction_full_response"] = pred.get("full_response", "")
            mapped_functions[matched_addr] = func_data
        else:
            unmatched_predictions.append(pred)
            print(f'[!] Warning: Could not match prediction for {function_id}')
    
    # Add unmatched Ghidra functions (without predictions)
    for addr, func_data in ghidra_data.items():
        if addr not in mapped_functions:
            func_data_copy = func_data.copy()
            func_data_copy["original_name"] = func_data.get("func_name", "")
            func_data_copy["predicted_name"] = ""
            func_data_copy["prediction_full_response"] = ""
            mapped_functions[addr] = func_data_copy
    
    print(f'[+] Matched {matched_count}/{len(predictions)} predictions')
    print(f'[+] Total functions in output: {len(mapped_functions)}')
    
    # Save mapped data
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(mapped_functions, f, indent=4)
    
    print(f'[+] Saved mapped functions to {output_path}')
    
    # Save unmatched predictions if any
    if unmatched_predictions:
        unmatched_file = output_path.replace('.json', '_unmatched.json')
        with open(unmatched_file, 'w') as f:
            json.dump(unmatched_predictions, f, indent=4)
        print(f'[!] Saved {len(unmatched_predictions)} unmatched predictions to {unmatched_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Map predictions to original Ghidra decompilation data'
    )
    parser.add_argument(
        '--ghidra_json', '-g',
        type=str,
        required=True,
        help='Path to original Ghidra JSON file (from decomp_for_stripped.py)'
    )
    parser.add_argument(
        '--predictions_json', '-p',
        type=str,
        required=True,
        help='Path to predictions JSON file (from batch_predict.py)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to save mapped functions JSON'
    )
    
    args = parser.parse_args()
    main(args.ghidra_json, args.predictions_json, args.output)

