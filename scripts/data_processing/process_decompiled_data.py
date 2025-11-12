import sys
from antlr4 import *
from antlr.CLexer import CLexer
from antlr.CParser import CParser
from antlr.CVisitor import CVisitor
import json
import os
import argparse

sys.setrecursionlimit(100000)

lines = []
lastLine = 0
lastOffset = 0

class MaskFunctionNameVisitor(CVisitor):
    def visitFunctionDefinition(self, ctx:CParser.FunctionDefinitionContext):
        if ctx.declarator() is not None:
            if ctx.declarator().directDeclarator() is not None:
                if ctx.declarator().directDeclarator().directDeclarator() is not None:
                    functionNameCtx = ctx.declarator().directDeclarator().directDeclarator()
                    start_token = functionNameCtx.start
                    stop_token = functionNameCtx.stop
                    lines[start_token.line - 1] = lines[start_token.line - 1][0: start_token.column] + '[MASK]' + lines[stop_token.line - 1][stop_token.column + len(stop_token.text) + lastOffset:]
        return self.visitChildren(ctx)


def process_dataset(unstripped_path, stripped_path, output_dir):
    global lines

    for root, dirs, files in os.walk(unstripped_path):
        for filename in files:
            print('[+] Now process ' + os.path.join(root, filename))

            py_dir = os.path.dirname(os.path.abspath(__file__))
            relative_path = os.path.relpath(root, unstripped_path)
            new_output_dir = os.path.normpath(os.path.join(py_dir, output_dir ,relative_path))
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir, exist_ok=True)

            d = {}
            
            with open(os.path.join(root, filename)) as f:
                unstripped_data = json.load(f)

            with open(os.path.join(stripped_path, relative_path, filename)) as f:
                stripped_data = json.load(f)

            for function_name in unstripped_data.keys():
                function = unstripped_data[function_name]
                start_address = function["function_address"]["start"]
                stripped_function = stripped_data[start_address]

                pair = {}

                if function is not None:
                    code = function["decomp_code"]

                    if function["assembly"] == ["?? ??"]:
                        continue

                    if len(function["assembly"]) > 510 or len(function["assembly"]) < 5:
                        continue

                    file = open('code.txt', 'w') # tmpfile
                    file.write(code)
                    file.close()

                    file = open('code.txt', 'r')
                    code = file.read()
                    antlrInput = InputStream(code)
                    file.close()

                    file = open('code.txt', 'r')
                    lines = file.readlines()
                    file.close()

                    lexer = CLexer(antlrInput)
                    stream = CommonTokenStream(lexer)
                    parser = CParser(stream)
                    tree = parser.compilationUnit()

                    visitor = MaskFunctionNameVisitor()
                    visitor.visit(tree)

                    res = ""
                    for line in lines:
                        res += line
                    pair["unstripped"] = res


                if stripped_function is not None:
                    stripped_code = stripped_function["decomp_code"]

                    if function["assembly"] == ["?? ??"]:
                        continue

                    # if len(function["assembly"]) > 510 or len(function["assembly"]) < 5:
                    #     continue

                    file = open('code.txt', 'w')
                    file.write(stripped_code)
                    file.close()

                    file = open('code.txt', 'r')
                    code = file.read()
                    antlrInput = InputStream(code)
                    file.close()

                    file = open('code.txt', 'r')
                    lines = file.readlines()
                    file.close()

                    lexer = CLexer(antlrInput)
                    stream = CommonTokenStream(lexer)
                    parser = CParser(stream)
                    tree = parser.compilationUnit()

                    visitor = MaskFunctionNameVisitor()
                    visitor.visit(tree)

                    res = ""
                    for line in lines:
                        res += line
                    pair["stripped"] = res
                    pair["stripped_function_name"] = stripped_function["func_name"]

                d[function_name] = pair

            with open(os.path.join(new_output_dir, filename), 'w') as f:
                print('[+] Write results to ' + os.path.join(new_output_dir, filename))
                json.dump(d, f, indent=4)


def process_binaries_for_prediction(stripped_path, output_dir):
    global lines

    for root, dirs, files in os.walk(stripped_path):
        for filename in files:
            print('[+] Now process', os.path.join(root, filename))

            py_dir = os.path.dirname(os.path.abspath(__file__))
            relative_path = os.path.relpath(root, stripped_path)
            new_output_dir = os.path.normpath(os.path.join(py_dir, output_dir ,relative_path))
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir, exist_ok=True)

            d = {}
            # print(os.path.join(root, filename))
            with open(os.path.join(root, filename)) as f:
                data = json.load(f)

            for function_name in data.keys():
                function = data[function_name]
                start_address = function["function_address"]["start"]

                pair = {}

                if function is not None:
                    code = function["decomp_code"]

                    file = open('code.txt', 'w')
                    file.write(code)
                    file.close()

                    file = open('code.txt', 'r')
                    code = file.read()
                    antlrInput = InputStream(code)
                    file.close()

                    file = open('code.txt', 'r')
                    lines = file.readlines()
                    file.close()

                    os.remove('code.txt')

                    lexer = CLexer(antlrInput)
                    stream = CommonTokenStream(lexer)
                    parser = CParser(stream)
                    tree = parser.compilationUnit()

                    visitor = MaskFunctionNameVisitor()
                    visitor.visit(tree)

                    res = ""
                    for line in lines:
                        res += line

                    d[function_name] = res
                
            with open(os.path.join(new_output_dir, filename), 'w') as f:
                print('[+] Write results to', os.path.join(new_output_dir, filename))
                json.dump(d, f, indent=4)


def main(args):
    if args.dataset == True and args.prediction == True:
        print("Error! You can just choose one mode '-d' or '-p'")
        sys.exit(0)

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.dataset:
        if not args.unstripped_path or not args.stripped_path:
            print("Error! Both --unstripped_path and --stripped_path are required for dataset mode (-d)")
            sys.exit(1)
        process_dataset(args.unstripped_path, args.stripped_path, output_dir)
    elif args.prediction:
        if not args.stripped_path:
            print("Error! --stripped_path is required for prediction mode (-p)")
            sys.exit(1)
        process_binaries_for_prediction(args.stripped_path, output_dir)
    else:
        print("Error! You should choose one mode '-d' or '-p'. Use '-h' to check.")
        sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine the stripped and unstripped decompiled code to generate the dataset, or output only the stripped decompiled code for prediction.')
    parser.add_argument('-d', '--dataset', action='store_true',
        help='Indicates the purpose of generating a new dataset for training and testing purposes.')
    parser.add_argument('-p', '--prediction', action='store_true',
        help='Indicates the purpose of prediction of a new stripped binary.')
    parser.add_argument('-u', '--unstripped_path', type=str, required=False,
        default=None,
        help="Path to JSON files containing decompiled unstripped binaries. Required for dataset mode (-d).")
    parser.add_argument('-s', '--stripped_path', type=str, required=False,
        default=None,
        help="Path to JSON files containing decompiled stripped binaries. Required for prediction mode (-p).")
    parser.add_argument('-o', '--output_dir', type=str, required=True,
        # default='',
        help='Directory to save the output files.')

    args = parser.parse_args()

    main(args)
