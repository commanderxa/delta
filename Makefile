MODE ?= dev

.PHONY: pybindings

pybindings:
ifeq ($(MODE),release)
	cd ./delta-py && maturin build --release
else
	cd ./delta-py && maturin develop --uv
endif