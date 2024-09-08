# Build the software, test it, generate the results and plots, and compile the
# manuscript PDF.
#
# Runs the individual Makefiles from code/ and manuscript/.

.PHONY: all
# Generate artifacts and render manuscript
all:
	make -C code all
	make -C manuscript all

.PHONY: clean
# Remove all artifacts from source code and manuscript
clean:
	make -C code clean
	make -C manuscript clean
