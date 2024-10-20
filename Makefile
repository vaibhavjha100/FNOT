.PHONY=run_builder run_inference install clean check runner_builder run_inference
.DEFAULT_GOAL=runner_inference

run_builder:
	@cd src && poetry run python runner_builder.py

run_inference:
	@cd src && poetry run python runner_inference.py

install: pyproject.toml
	poetry install

clean:
	poetry run python -c "import os; import shutil; [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__']"

check:
	poetry run flake8

runner_builder: install run_builder clean

runner_inference: install run_inference clean