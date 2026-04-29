py_develop:
	cd ./delta-py && maturin develop --uv

py_release:
	cd ./delta-py && maturin build --release
