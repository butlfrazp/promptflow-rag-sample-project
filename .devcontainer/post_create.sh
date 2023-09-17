#!/bin/bash

# install the packages from requirements.txt
conda run -n conda-env pip install -r /workspaces/promptflow-custom-tool/src/requirements_dev.txt

# Add CORS fix config for jupyter-notebook
cp ~/.jupyter/jupyter_server_config.py ~/.jupyter/jupyter_notebook_config.py

# Mark environment as initialized
bash /workspaces/promptflow-custom-tool/.devcontainer/scripts/mark_initialized.sh
