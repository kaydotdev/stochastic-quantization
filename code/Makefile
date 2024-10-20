# Manages building, testing, and producing results and figures for the paper.


.PHONY: all
all: install test results

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

.PHONY: results
# Run Jupyter Notebook and save produced figures and other artifacts
results:
	jupyter nbconvert --to markdown --execute ./notebooks/quantization.ipynb
	@rm ./notebooks/quantization.md
	@if [ ! -d ../results/figures ]; then \
		mkdir -p ../results/figures; \
	fi
	@( \
		NB_SOURCE_FILENAMES="quantization_9_0.png quantization_19_0.png quantization_21_0.png quantization_27_0.png"; \
		NB_RESULT_FILENAMES="dataset.png latent_space.png sq_convergence_full_dataset.png sq_quants.png"; \
		i=1; \
		for src in $$NB_SOURCE_FILENAMES; do \
			res=$$(echo $$NB_RESULT_FILENAMES | cut -d' ' -f$$i); \
			mv "./notebooks/quantization_files/$$src" "../results/figures/$$res"; \
			i=$$((i + 1)); \
		done; \
	)
	@rm -rf ./notebooks/quantization_files
