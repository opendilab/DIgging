# DIgging

DIgging -- Decision Intelligence for digging better parameters in target function/environments.


## Introduction

**DIgging** is a heuristic searching and optimization platform with various Decision Intelligence methods such as Genetic Algorithm, Bayesian Optimization and Reinforcement Learning etc. It can be used to digging better candidates to handle combinatorial optimization problems and non-gradient search problems.

**DIgging** is a fundamental platform under [**OpenDILab**](http://opendilab.org/) and it uses [**DI-engine**](https://github.com/opendilab/DI-engine) to build RL searching pipelines.

[Documentation](https://opendilab.github.io/DIgging/index.html)


## Outlines
- [DIgging](#digging)
  - [Introduction](#introduction)
  - [Outlines](#outlines)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [License](#license)


## Installation

You can simply install DIgging with `pip` command line from the official PyPI site.

```bash
pip install --user digging
python -c 'import digging'
```

Or you can simply install it from the source code.

```bash
git clone git clone https://github.com/opendilab/DIgging.git
cd DIgging
pip install --user .
python -c 'import digging'
```

It will automatically install **DI-engine** together with its requirement packages i.e. **PyTorch**.

## Quick start

## License

`DIgging` released under the Apache 2.0 license.
