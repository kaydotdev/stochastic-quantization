# Manages building and testing of the source code.


.PHONY: all
all: install test

.PHONY: clean
# Remove all processing artifacts, build files and cache files
clean:
	find . -name "*.pyc" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache */.ipynb_checkpoints

.PHONY: install
# Build `sq` package and install in the local environment
install:
	pip install --no-deps -e .

.PHONY: test
# Run the test suite with built-in 'unittest' package
test:
	python -m unittest discover -s tests -p "test_*.py"
