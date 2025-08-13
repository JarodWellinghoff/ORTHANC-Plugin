#!/usr/bin/env python3
import os
from glob import glob, escape
import shutil

def print_tree(root, exclude=None, move=None, out=print):
    def _filter_excluded(d, exclude):
        # Recursively filter out excluded folders
        for key in list(d.keys()):
            if key in exclude:
                if len(d[key]) > 0:
                    d[key] = '...'
                else:
                    del d[key]
            else:
                # If the key is not excluded, recurse into the directory
                if len(d[key]) > 0:
                    _filter_excluded(d[key], exclude)
                else:
                    d[key] = 'EOF'
    def _print_tree(d, prefix="", out=out):
        items = list(d.items())
        last_index = len(items) - 1
        for i, (key, value) in enumerate(items):
            connector = "└───" if i == last_index else "├───"
            if isinstance(value, dict):
                out(f"{prefix}{connector}{key}/")
                extension = "    " if i == last_index else "│   "
                _print_tree(value, prefix + extension)
            elif value == "...":
                out(f"{prefix}{connector}{key}/")
                extension = "    " if i == last_index else "│   "
                out(f"{prefix + extension}└───...")
            else:
                out(f"{prefix}{connector}{key}")
    def _move_files(files, out=out):
        if not files:
            print("No files to move.")
            return
        if os.listdir(move):
            # Empty the move directory there are files in it
            for f in os.listdir(move):
                try:
                    os.remove(os.path.join(move, f))
                except Exception as e:
                    print(f"Error removing {f}: {e}")
        for f in files:
            try:
                new_path = os.path.join(root, move, os.path.basename(f))
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy2(f, new_path)
                print(f"Copied: {f} -> {new_path}")
            except Exception as e:
                print(f"Error copying {f}: {e}")

    def _tree(dir_path, prefix=''):
        all_files_and_dirs = sorted(glob(os.path.join(escape(root), "**", "*"), recursive=True, include_hidden=True))
        files_and_dirs_to_remove = []
        for f in all_files_and_dirs:
            for ex in exclude:
                if ex in f.split(os.sep):
                    files_and_dirs_to_remove.append(f)
        filtered_files_and_dirs = sorted(list(set(all_files_and_dirs) - set(files_and_dirs_to_remove)))
        filtered_files = [f for f in filtered_files_and_dirs if os.path.isfile(f)]
        asdf1 = [os.path.relpath(f, root) for f in filtered_files_and_dirs]
        asfd2 = [f.split(os.sep) for f in asdf1]
        asdf3 = {}
        for f in asfd2:
            d = asdf3
            for part in f:
                d = d.setdefault(part, {})
        # filter out excluded folder(s)
        _filter_excluded(asdf3, exclude)
        _print_tree(asdf3, out=out)
        _move_files(filtered_files, out=out)

    # display root as drive letter + '.' on Windows, else '.'
    # drive, _ = os.path.splitdrive(root)
    # root_label = drive + '.' if drive else '.'
    out(root)
    _tree(root)

def main():
    # parser = argparse.ArgumentParser(description="Print a tree of the current directory.")
    # parser.add_argument('-e', '--exclude',
    #                     help="Folder (relative to cwd) to exclude (and its sub-dirs)")
    # parser.add_argument('-o', '--output',
    #                     help="Path to save tree to a txt file")
    # args = parser.parse_args()

    output = 'tree.txt'
    move = 'flatten'
    exclude = [
        '.venv', 
        '__pycache__', 
        '.vscode', 
        'OLD', 
        'certificates', 
        'data', 
        move, 
        'minio-data', 
        'Dockerfile', 
        'CHO_Calculation_Patient_Specific_skimage_Canny_edge_v12.py',
        'CHO_Calculation_Patient_Specific_skimage_Canny_edge_v13.py',
        'CHO_Calculation_Patient_Specific_skimage_Canny_edge_Global_Noise.py',
        'requirements.txt', 
        'orthanc.pyi', 
        'orthanc-native.json', 
        '.env', 
        '.dockerignore', 
        'print_tree.py',
        'frontend'
    ]

    cwd = os.getcwd()
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            print_tree(cwd, exclude=exclude, move=move, out=lambda s: f.write(s + '\n'))
    else:
        print_tree(cwd, exclude=exclude, move=move)

if __name__ == '__main__':
    main()
