# kaggle-connectx-competition
Train Reinforcement Learning agents to play in the ConnectX Kaggle competition

# Installation

Poetry is not used due to dependency issues between kaggle-environments, shimmy and gym.
Tested on M2 Mac with python 3.10.12 with the following:
```
pip install setuptools==65.5.0 pip==21
pip install numpy kaggle kaggle-environments==1.12.0 wandb 'stable-baselines3[extra]' rl-zoo3 opencv-python moviepy jupyterlab notebook gym ipywidgets
```

See https://stackoverflow.com/questions/77124879/pip-extras-require-must-be-a-dictionary-whose-values-are-strings-or-lists-of for why we are downgrading pip




# Dev installation
```
pip install ruff pre-commit
pre-commit install
pre-commit run --all-files
```