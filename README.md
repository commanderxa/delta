
<div align="center">
  <img src="media/header.png" width="100%" alt="Project Header">
</div>

<br>

<h1 align="left">The <b>ATHENA</b> Project</h1>

<div align="left">
   <a href="https://github.com/commanderxa/athena">
      <img src="https://img.shields.io/badge/Rust-1.95.0%2B-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust 1.95.0+">
   </a>
   <a href="https://github.com/commanderxa/athena/actions">
      <img src="https://img.shields.io/github/actions/workflow/status/commanderxa/athena/rust.yml?branch=master&style=for-the-badge&logo=githubactions&logoColor=white&label=Build" alt="Build">
   </a>
   <a href="https://github.com/commanderxa/athena/stargazers">
      <img src="https://img.shields.io/github/stars/commanderxa/athena?style=for-the-badge&logo=github&logoColor=white&label=Stars" alt="Stars">
   </a>
   <a href="https://github.com/commanderxa/athena/commits/master">
      <img src="https://img.shields.io/github/last-commit/commanderxa/athena?style=for-the-badge&logo=git&logoColor=white&label=Last%20Commit" alt="Last Commit">
   </a>
   <a href="https://github.com/commanderxa/athena/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/commanderxa/athena?style=for-the-badge&logo=opensourceinitiative&logoColor=white&label=License" alt="License">
   </a>
</div>

<br>

A Deep Learning library implemented in pure `Rust`. It is inspired by [`micrograd`](https://github.com/karpathy/micrograd) project.

## Setup

This project requires the version of `Rust` $>= 1.95$.

To build the library run:

```sh
cargo build
```

Ensure that no test fails:

```sh
cargo test
```

## Usage

See `/examples` for usage examples. To run an example, execute the following line passing a file name:

```sh
cargo run --example <name>
```

## License

This project is licensed under the terms of Apache 2.0 license.
See the [LICENSE](./LICENSE) file for details.
