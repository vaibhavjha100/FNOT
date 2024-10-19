.PHONY=run install clean check runner
.DEFAULT_GOAL=runner

run:
	@cd src && poetry run python runner.py

install: pyproject.toml
	poetry install

clean:
	poetry run python -c "import os; import shutil; [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__']"

check:
	poetry run flake8

runner: install run clean