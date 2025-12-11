# Training Environment (PC-Side)

This directory contains the **machine learning training pipeline** for the  
ESP32 Object/Distance Recognition Project.  
It is fully managed using **uv**, with deterministic environments via:

- `pyproject.toml` → declares the project's dependencies and metadata  
- `uv.lock` → pins exact package versions for reproducible setups  

If you are a new developer joining the project, follow the steps below.

```powershell
# Install uv (if not installed)
pip install uv

# Go to training environment directory path
cd ./training_pc

# Sync the environment (creates venv + installs deps)
uv sync

# Run Jupyter Notebook or Lab
uv run jupyter lab
uv run jupyter notebook

# activate .venv
.\.venv\Scripts\activate

## or you can run train
uv run python src/train.py

# If anything breaks, simply run
uv clean
uv sync
```

You're ready to train!

## Use `pyproject.toml`

```powershell
# if you don't have `pyproject.toml`
uv init .

# add packages. for example 
uv add numpy pandas matplotlib scikit-learn jupyter

# create .venv and sync & installs all dependencies exactly as specified in `uv.lock`
uv sync

# run train
uv run python src/train.py
```

### to add new packages

```powershell
uv add jupyter notebook
uv sync
```

## Use `uv.lock`
