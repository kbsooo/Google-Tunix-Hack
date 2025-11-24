import json

notebook_path = '/Users/kbsoo/Codes/kaggle/tunix-hack/tunix_training_complete.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Add nest_asyncio to imports (Cell 3)
cell3_source = nb['cells'][2]['source']
if isinstance(cell3_source, str):
    cell3_source = [cell3_source]
new_imports = [
    "import nest_asyncio\n",
    "nest_asyncio.apply()\n"
]
# Insert at the beginning
nb['cells'][2]['source'] = new_imports + cell3_source

# 2. Modify Configuration (Cell 4) to be dynamic
cell4_source = nb['cells'][3]['source']
# Find where MESH is defined and replace it with dynamic logic
new_config_logic = [
    "# ====== Environment Detection ======\n",
    "try:\n",
    "    # Try to detect TPU\n",
    "    jax.distributed.initialize()\n",
    "    devices = jax.devices()\n",
    "    print(f\"Detected {len(devices)} devices: {devices}\")\n",
    "except:\n",
    "    print(\"TPU initialization failed or not available. Falling back to local devices.\")\n",
    "    devices = jax.devices()\n",
    "    print(f\"Fallback devices: {devices}\")\n",
    "\n",
    "# ====== Sharding ======\n",
    "if len(devices) >= 4:\n",
    "    MESH = [(1, 4), (\"fsdp\", \"tp\")]\n",
    "else:\n",
    "    # Fallback for local/CPU/Single-GPU\n",
    "    print(f\"Warning: Only {len(devices)} devices found. Using single-device mesh.\")\n",
    "    MESH = [(1, 1), (\"fsdp\", \"tp\")]\n",
    "\n"
]

# Replace the static MESH definition
new_source = []
for line in cell4_source:
    if "MESH = [(1, 4)" in line:
        new_source.extend(new_config_logic)
    else:
        new_source.append(line)

nb['cells'][3]['source'] = new_source

# 3. Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
