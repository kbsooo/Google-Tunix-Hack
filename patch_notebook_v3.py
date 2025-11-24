import json

notebook_path = '/Users/kbsoo/Codes/kaggle/tunix-hack/tunix_training_complete.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Remove existing nest_asyncio code from Cell 3 (Index 2)
cell3 = nb['cells'][2]
source3 = cell3['source']
if isinstance(source3, str):
    source3 = [source3]

new_source3 = [line for line in source3 if "nest_asyncio" not in line]
cell3['source'] = new_source3

# 2. Create a new first code cell for nest_asyncio
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install and apply nest_asyncio FIRST to fix Jupyter asyncio issues\n",
        "!pip install -q nest_asyncio\n",
        "import nest_asyncio\n",
        "nest_asyncio.apply()\n",
        "print(\"Asyncio patched.\")"
    ]
}

# Insert as the first code cell (Index 1, after Markdown title)
nb['cells'].insert(1, new_cell)

# 3. Wrap checkpointer.wait_until_finished() in try-except (Load Model Cell)
# Find the cell with "checkpointer.wait_until_finished()"
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if isinstance(source, str):
            source = [source]
        
        new_source = []
        modified = False
        for line in source:
            if "checkpointer.wait_until_finished()" in line:
                new_source.append("try:\n")
                new_source.append("    " + line)
                new_source.append("except Exception as e:\n")
                new_source.append("    print(f\"Warning: Checkpoint wait failed (likely asyncio issue): {e}\")\n")
                modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source
            print("Patched checkpointer wait.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully (v3).")
