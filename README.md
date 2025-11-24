# Model-Based Reinforcement Learning for Energy Management in Smart Buildings

This repository contains implementations of model-based and model-free reinforcement learning algorithms applied to the CityLearn environment for smart building energy management. It includes the CHECA algorithm (CityLearn Challenge 2023 winner) and several model-based RL baselines.

---

## 🔧 Installation

Create the conda environment:

```bash
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yaml
```

---

## 📁 Project Structure

```
docs/                     # Reference papers for CityLearn and algorithms
src/
 ├── rewards/             # Reward functions for the CityLearn environment
 └── agents/
      ├── checa/          # CHECA algorithm implementation
      └── model_based/    # Model-based RL algorithms (MACURA, SAC, M2AC, MBPO)
```

---

## 🚀 Running Experiments

All algorithms use the same environment setup.

### Activate Environment

Before running any experiment:

```bash
conda deactivate
conda activate macura_env_gymnasium_hpc_compatible
```

---

## 🟦 CHECA

```bash
cd src
python -m agents.checa.main
```

---

## 🟩 Model-Based RL Algorithms

Navigate to the model-based directory:

```bash
cd src/agents/model_based
```

Then run the desired algorithm:

### MACURA

```bash
python -m mbrl.examples.main --config-name=launcher_macura
```

### SAC

```bash
python -m mbrl.examples.main --config-name=launcher_sac
```

### M2AC

```bash
python -m mbrl.examples.main --config-name=launcher_m2ac
```

### MBPO

```bash
python -m mbrl.examples.main --config-name=launcher_mbpo
```

## ⚙️How to change parameters

> Follow the file path below 
```PATH: src/agents/model_based/mbrl/example/```

> Here you can find all the `.yaml` configuration files.  

The **launcher file** defines train and test config
The **main files** define the connection between the various components.

-   The `**overrides**` folder contains **environment-specific parameters**.
    
-   The `**dynamics_model**` folder includes the configuration for the **probabilistic neural networks (PNNs)**.
    
-   The `**algorithm**` folder contains parameters for each specific algorithm (e.g., MACURA, MBPO, M2AC).
    