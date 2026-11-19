PYTHON ?= python3

.PHONY: install test smoke build
install:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) -m resnet_smote smoke
build:
	$(PYTHON) -m build
