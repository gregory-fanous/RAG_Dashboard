# conda create -n RAGD python=3.10 -y
# conda activate RAGD
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade build
pip install pip-tools
pip-compile pyproject.toml
pip install -r requirements.txt

