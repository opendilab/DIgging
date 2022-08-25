# DIgging

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
[![Code Style](https://github.com/opendilab/digging/actions/workflows/style.yml/badge.svg)]()
[![Code Test](https://github.com/opendilab/digging/actions/workflows/test.yml/badge.svg)](https://github.com/opendilab/DIgging/actions?query=workflow%3A%22Code+Test%22)
[![Docs deploy](https://github.com/opendilab/digging/actions/workflows/doc.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/doc.yml?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/HansBug/hbayes/workflows/Package%20Release/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/release.yml?query=workflow%3A%22Code+Test%22)
[![codecov](https://img.shields.io/codecov/c/github/opendilab/digging)](https://img.shields.io/codecov/c/github/opendilab/digging)

![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/digging)](https://github.com/opendilan/digging/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/digging)](https://github.com/opendilab/digging/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/digging)
[![GitHub license](https://img.shields.io/github/license/opendilab/digging)](https://github.com/opendilab/digging/blob/master/LICENSE)


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
