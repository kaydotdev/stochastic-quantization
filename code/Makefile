# Manages building, testing, and cleaning the code as well as running the code
# to generate the results and figures for the paper.


.PHONY: all
all: build test

.PHONY: clean
# Remove all processing artifacts, build files and cache files
clean:
	find . -name "*.pyc" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache */.ipynb_checkpoints

.PHONY: build
# Build `sq` package and install in the local environment
build:
	pip install --no-deps -e .

.PHONY: test
# Run the test suite with built-in 'unittest' package
test:
	python -m unittest tests/test_calculate_loss.py
	python -m unittest tests/test_find_nearest_element.py
