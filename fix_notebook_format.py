import json

notebook_path = '/Users/kbsoo/Codes/kaggle/tunix-hack/tunix_training_complete.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = cell.get('source', [])
    
    # Check if source is a list of characters (mostly length 1)
    if isinstance(source, list) and len(source) > 0:
        # Heuristic: if average length is close to 1, it's likely broken
        avg_len = sum(len(s) for s in source) / len(source)
        if avg_len < 1.5:  # Threshold
            print(f"Fixing cell {cell.get('id', 'unknown')}...")
            full_text = "".join(source)
            # Convert back to list of lines
            new_source = full_text.splitlines(keepends=True)
            cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook format fixed.")
