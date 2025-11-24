import json

notebook_path = '/Users/kbsoo/Codes/kaggle/tunix-hack/tunix_training_complete.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Add nest_asyncio to imports (Cell 3 - Index 2)
# Check if nest_asyncio is already there
cell_imports = nb['cells'][2]
source_imports = cell_imports['source']
if isinstance(source_imports, str):
    source_imports = [source_imports]

if not any("nest_asyncio" in line for line in source_imports):
    print("Adding nest_asyncio...")
    new_imports = [
        "!pip install -q nest_asyncio\n",
        "import nest_asyncio\n",
        "nest_asyncio.apply()\n"
    ]
    cell_imports['source'] = new_imports + source_imports

# 2. Robust TPU Initialization (Cell 4 - Index 3)
cell_config = nb['cells'][3]
source_config = cell_config['source']
if isinstance(source_config, str):
    source_config = [source_config]

# Replace the static MESH definition with dynamic logic
new_config_logic = [
    "# ====== Environment Detection ======\n",
    "try:\n",
    "    # Try to detect TPU\n",
    "    import jax\n",
    "    jax.distributed.initialize()\n",
    "    devices = jax.devices()\n",
    "    print(f\"Detected {len(devices)} devices: {devices}\")\n",
    "except Exception as e:\n",
    "    print(f\"TPU initialization failed: {e}\")\n",
    "    print(\"Falling back to local devices.\")\n",
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

# Find where MESH is defined and replace it
new_source = []
mesh_replaced = False
for line in source_config:
    if "MESH = [(1, 4)" in line:
        if not mesh_replaced:
            new_source.extend(new_config_logic)
            mesh_replaced = True
    else:
        new_source.append(line)

if not mesh_replaced:
    # If exact line not found, maybe it was already patched? 
    # Or maybe format is different. Just append if not found?
    # Better to be safe and check if "Environment Detection" is already there
    if not any("Environment Detection" in line for line in source_config):
        print("Injecting Environment Detection logic...")
        # Inject before the first config line
        new_source = new_config_logic + source_config
    else:
        new_source = source_config

cell_config['source'] = new_source

# 3. Save
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully (v2).")
