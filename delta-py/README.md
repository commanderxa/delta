# DELTA Python

Official bindings of **DELTA** Deep Learning library from `rust` to `Python`. `pyo3` library is used to create the bindings. 

## Setup

This project requires the version of `Rust` >= 1.95 as well as `Python` package manager `uv` installed.

Pull dependencies:

```sh
uv sync
```

> Make sure that you activated the virtual environment afterwards to run the next commands.


To generate the bindings for release run (in the root folder [`/delta`](../delta/)):

```sh
make py_release
```

For development:

```sh
make py_develop
```


## Installation

In order to install a python package locally (with `uv`), run (in the [project root](../delta/)):

```sh
uv add <PATH_TO_THE_PROJECT_ROOT>/target/wheels/delta-0.1.0-*.whl
```

> Replace `<PATH_TO_THE_PROJECT_ROOT>` with the actual path.

## License

This project is licensed under the terms of Apache 2.0 license.
See the [LICENSE](../LICENSE) file for details.
