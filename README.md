
# 🔬 Model-Based RL + CHESCA Simulation Toolkit

This project contains a collection of Jupyter Notebooks and Python modules for training and evaluating **Model-Based Reinforcement Learning (MBRL)** algorithms such as **MACURA**, **MBPO**, and **M2AC**, and for simulating a real-world energy environment with the **CHESCA** platform (built on CityLearn). This README is made up to explain how to run on Google Colab. Refer to the **pdf** for more theoretical basis.

---

## 📁 Project Structure

src/  
├── MACURASimulation.ipynb <- Run MACURA-based MBRL training/evaluation  ( as example for others) 

├── CHESCASimulation.ipynb <- Run CHESCA energy environment simulation  
├── agents/ <- CHESCA and MBRL algorithms code
├── wrappers/ <- CityLearn wrappers  
├── rewards/ <- Custom energy-based reward functions  
├── utils/ <- Helper code (mainly plotting)  
└── old/ <- Old Jupyter Notebooks with previous test

## 📦 Run on Google Colab

> ✅ You can run this entire simulation on [Google Colab](https://colab.research.google.com/) with no local installation required. All the already present Jupyter Notebooks are set to be run on Google Colab.

### 🔗 Donwload Setup (Colab)

> Upload these files directly by downloading them from GitHub and manually placing them into your [Google Drive](https://drive.google.com/) folder. Open a new notebook  and go on.

> Or open a new Google Colab notebook and run this code in the first cell. This code must be temporary and used only to download the first time.
```python
from google.colab import drive
drive.mount('/content/drive')   # Mount your Google Drive

%cd /content/drive/MyDrive/		# Select the directory in which you want to download
!git clone https://github.com/andreablushi/Model-Based-Reinforcement-Learning-for-energy-management-in-smart-buildings	# Git Clone
```
### 🧠 Execute on Colab
> Those steps must be done everytime you execute in Colab, no matters the case.
---
### Dependecies
> 🔴​ Every time you run the code, you'll need to **manually install all dependencies**, adjusting the versions of all libraries. You might see a popup asking you to restart the execution; if so, **do not touch the popup** and wait for the installation to finish. Once it's done, restart the execution. You won't need to run that cell again until you close Google Colab.

> Those are the dependecies for **MBRL Algorithm**
```bash
# Forces NumPy TensorFlow to be compatibile with CityLearn
!pip uninstall -y numpy tensorboard tensorflow
!pip install numpy==1.23.5
!pip install tensorflow==2.12.0
!pip install tensorboard==2.12.3
!pip install hydra-core
!pip install citylearn
!pip install omegaconf
!pip install colorednoise
!pip install mujoco
```
>Those are for **CHESCA** or CityLearn build-in algorithms
```bash
# Forces NumPy TensorFlow to be compatibile with CityLearn
!pip uninstall -y numpy tensorboard tensorflow
!pip install numpy==1.23.5
!pip install tensorflow==2.12.0
!pip install tensorboard==2.12.3
!pip install citylearn
```

---
### File Mount
>To load files and enable your Colab Notebooks to interact with your file structure, you **always have to mount your Google Drive.** 
```python
from google.colab import drive
drive.mount('/content/drive')   # Mount your Google Drive
```

>It might be necessary to modify the working directory in order to correctly resolve dependencies. We'll provide an example on how to handle this issue.
```python
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/src')

# Various basic import ...

# Loads correctly MBRL. This is important also for local execution.
import os
os.chdir('/content/drive/MyDrive/Colab Notebooks/src/agents/model_based')

# MBRL import
```

## 📦 Run on local

>When you're running the code locally, you'll need to **download the previously mentioned dependencies**. After that, you should be able to run it without issues. You might also need to **install basic libraries like Pandas and Matplotlib**, as these are already pre-installed in Google Colab.


## 🧠 Run MACURA (Model-Based RL)

> Now, we will demonstrate the code's functionality regarding the execution of the MBRL algorithms we discussed, along with SAC, both originating from [MACURA](https://github.com/Data-Science-in-Mechanical-Engineering/macura) developers' implementation.

>Initially, we need to handle the imports, following the management style mentioned previously.
```python
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/src')
# CityLearn Package
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import *
from citylearn.wrappers import *

# Data manipulator or plotter
from utils import plotting_functions as plt
import pandas as pd
from IPython.display import display, Markdown
from hydra import initialize, compose
import numpy as np
import omegaconf
import torch

import os
os.chdir('/content/drive/MyDrive/Colab Notebooks/src/agents/model_based')
  
# Here you import the chosen agent
import agents.model_based.mbrl.algorithms.macura as macura
# Package for the env handler
import agents.model_based.mbrl.util.env as env_util

# Import to clear the config instance
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
```

> Once all imports are resolved, we can focus on starting with the loading of the algorithms' configuration files. It functions **correctly** if the referenced folder relocation was performed in the imports. It is important to specify the correct config file for test and training, or you will see uncorrect perfomance. I suggest to print a configuration parameter to see if correctly uploaded and if it's sincronized with your last changes

```python
# Specify the configuration path
initialize(config_path="./mbrl/examples/conf")

# Load the main configuration file
cfg = compose(config_name="main_macura") 

# Load the configuration file for testing
test_cfg = compose(config_name="test_macura") 

# Test
print(cfg.overrides.num_steps)

# This line of code is optional. It cleans the previous execution results.
[shutil.rmtree(p) if os.path.isdir(p) else os.unlink(p) for p in [os.path.join('./exp/macura', f) for f in os.listdir('./exp/macura')]]
```
>You might notice a slightly different execution in the uploaded files compared to what we're about to describe, but it will be **equivalent**. It's up to the **developer's discretion** which of the two approaches to use. The first thing to do is to load the environment for training and evaluate.
> `distance_env`  is  a **typo left by the MACURA developers**, and it's actually never called as a parameter in the code. If preferred, you can directly load `env` or `test_env`.
```python
print(f"Using the following algorithm: {cfg.algorithm.name}!")

# Setting random seed in NumPy and Torch
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# Training env
env, term_fn, reward_fn = env_util.EnvHandler.make_env(cfg, test_env=False)
# It's useless.
distance_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
print(f"Using the following reward function: {reward_fn.__class__.__name__}")
# Evaluate env
test_env, *_ = env_util.EnvHandler.make_env(test_cfg, test_env=True)
```
>Taking `env` as an example, we need to load its corresponding configuration file and declare whether it's the `test_env` using a boolean flag. The `env_util.EnvHandler` function will return the environment, along with the termination and reward functions. We are exclusively interested in the environment; for CityLearn, the termination function is None. We will then see how this is managed.

> You can now **execute the code** using the `train` function. It requires the **previously calculated data** and a **relative path** for loading the execution results. Environment results are stored in `test_env`. If you want to see the building's performance at each step of the last evaluation episode, refer to this variable. It also holds the resulting values of the **Cost Functions**, which are used as a comparison index and described in [CityLearn documentation ](https://www.citylearn.net/overview/cost_function.html).
```python
macura.train(env, test_env,distance_env, term_fn, cfg, work_dir="./exp/macura")
```
> To view and utilize the various plotting functionalities, you should use an `.ipynb` file as an example to better understand what data is plotted and how. For additional parameters, refer to the [CityLearn documentation ](https://www.citylearn.net/overview/observations.html), which shows the observation data.

>Each **experiment** saves under `agents/model_based/exp/{macura,mbpo,m2ac}/`, and for all plotting function you can turn on the `save` flag to download the plot results.

## ⚙️How to load a new environment in MBRL
> If you intend to **modify the dataset to load**, the **environment configuration**, or the **reward function**, follow these steps.

> Follow the file path below and open the following Python file: `env.py`. Then, locate the `_legacy_make_env` function.
```PATH: agents/model_based/mbrl/```**util**```/env.py```

> You will find a code structured like this:
```python
# ....... List of other Mujoco enviroment ...............
if  cfg.overrides.env ==  "humanoid_truncated_obs":
	env  =  mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv(render_mode=render_mode)
	term_fn  =  mbrl.env.termination_fns.humanoid
	reward_fn  =  None

# Training CityLearn
elif  cfg.overrides.env ==  "citylearn":

	# Here you can change the dataset name, you can't change central agent for base MBRL
	env  = CityLearnEnv("citylearn_challenge_2023_phase_2_local_evaluation", central_agent=True)
	
	# Here you can change the reward function
	env.reward_function = SolarPenaltyAndComfortReward(env.schema)
	# Here is not relevant, you can leave it none. If you use models with known reward it's relevant.
	reward_fn  = SolarPenaltyAndComfortReward(env.schema)
	# Suggested wrapper
	env  = NormalizedSpaceWrapper(env)
	# Mandatory wrapper
	env  = StableBaselines3Wrapper(env)
	# Left it to default
	term_fn  =  mbrl.env.termination_fns.no_termination

# Test set CityLearn
elif  cfg.overrides.env ==  "test_citylearn":
	env  = CityLearnEnv("citylearn_challenge_2023_phase_2_online_evaluation_3", central_agent=True)
	env.reward_function = SolarPenaltyAndComfortReward(env.schema)
	reward_fn  = SolarPenaltyAndComfortReward(env.schema)
	env  = NormalizedSpaceWrapper(env)
	env  = StableBaselines3Wrapper(env)
	term_fn  =  mbrl.env.termination_fns.no_termination
```
>**To add a new environment, simply define a new `elif` block inside the `legacy_make_env` function.**  
In this block: assign a **unique name** (which will be used in the YAML configuration), **create and initialize** the environment, **specify** the reward function and termination function.
It is **recommended** to normalize the environment (if applicable) and wrap it to be **Gym-compatible**.

## ⚙️How to change parameters

> Follow the file path below 
```PATH: agents/model_based/mbrl/example/```

> Here you can find all the `.yaml` configuration files.  
The **main files** define the connection between the various components and are the ones referenced in the previous code.

-   The `**overrides**` folder contains **environment-specific parameters**.
    
-   The `**dynamics_model**` folder includes the configuration for the **probabilistic neural networks (PNNs)**.
    
-   The `**algorithm**` folder contains parameters for each specific algorithm (e.g., MACURA, MBPO, M2AC).
    

For more detailed information, please refer to the `Parameters_Tutorial.yaml`file provided directly by the [MACURA]((https://github.com/Data-Science-in-Mechanical-Engineering/macura/blob/master/How_to_start_experiments/Config_files_explained.yaml)) developers.

---
## 🧠 Run CHESCA 

> We starts from import required modules
```python
# CityLearn environment and wrappers
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import *

# CHESCA agent
from agents.checa.agent import Checa as Agent

# Reward functions
from rewards.CityLearnReward import SolarPenaltyAndComfortReward
from rewards.ComfortandConsumptionReductionReward import ComfortandConsumptionReductionReward

# Data & plotting
import pandas as pd
from datetime import datetime
from utils.plotting_functions import create_episode_table, plot_kpis
from IPython.display import display, Markdown

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
```

### 🔍 Initialize training and evaluation environments

> Set up two CityLearn environments: one for training, one for evaluation, and assign a common reward function. It's very similar to the setting in MBRL. This is mandatory for all CityLearn's implemented algorithm.
```python
# Create two CityLearnEnv instances (local & online evaluation)
train_env = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation', 
    central_agent=True
)
eval_env  = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_online_evaluation_3', 
    central_agent=True
)

# Instantiate and assign the reward function
reward_fn = SolarPenaltyAndComfortReward(train_env.schema)
train_env.reward_function = reward_fn
eval_env.reward_function  = reward_fn

# (Optional) set deterministic seeds for reproducibility
train_env.random_seed = 0
eval_env.random_seed  = 0

```

###  🔍Instantiate the CHESCA agent
> Create the CHESCA agent, specifying how many episodes to train and the building considered for the plotting
```python
# Agent
model = Agent(train_env)

# Number of the chosen building and number of episode trainig
num_building = 0
num_episodes = 1
```
### 🔍Train the agent

> Run the learning loop and collect training rewards for analysis.
```python
# Reset the environment
observations, _ = train_env.reset()
# Run training
model.learn(
    episodes=num_episodes, 
    deterministic_finish=False
)
# Display training results
display(Markdown("### Training Episode Rewards"))
# Using plotting function
df_train = create_episode_table(train_env.episode_rewards)
display(df_train)
```

### 🔍Evaluate the agent

> Due to CityLearn limitations, it is **mandatory** to perform the evaluation **manually**, as automated evaluation hooks are not supported so easly.

```python
# Reset evaluation environment
observations, _ = eval_env.reset()

# Step through until termination
while not eval_env.terminated:
    actions = model.predict(observations, deterministic=True)
    observations, _, _, _, _ = eval_env.step(actions)
```

>  This manual training and evaluation procedure applies to all implemented algorithms from CityLearn ( for furthe information consults CityLearn [doc](https://www.citylearn.net/quickstart.html)), and the plotting workflow remains identical to the previous notebooks, with the exception that MBRL-specific metrics are not automatically included.  
