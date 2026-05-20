MODE ?= dev

.PHONY: pybindings

pybindings:
ifeq ($(MODE),release)
	cd ./crates/delta-py && maturin build --out ../../target/wheels/ --release
else
	cd ./crates/delta-py && maturin develop --uv
endif