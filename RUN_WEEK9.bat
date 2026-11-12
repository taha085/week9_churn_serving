@echo off
python -m pip install -r requirements.txt
python src\serialize_model.py
python src\log_experiment.py
pytest -q
