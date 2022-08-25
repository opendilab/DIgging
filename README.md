# DIgging

<img src="./docs/figs/digging_banner.png" alt="icon"/>

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
[![Style](https://github.com/opendilab/digging/actions/workflows/style.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/style.yml?query=workflow%3A%22Style+And+Badge%22)
[![Test](https://github.com/opendilab/digging/actions/workflows/test.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/test.yml?query=workflow%3A%22Code+Test%22)
[![Docs](https://github.com/opendilab/digging/actions/workflows/doc.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/doc.yml?query=workflow%3A%22Docs+Deploy%22)
[![Package](https://github.com/opendilab/digging/actions/workflows/release.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/release.yml?query=workflow%3A%22Package+Release%22)
[![codecov](https://img.shields.io/codecov/c/github/opendilab/digging)](https://app.codecov.io/gh/opendilab/digging)
![Loc](https://img.shields.io/endpoint?url=https://gist.github.com/RobinC94/7f38f27fb3b34d4bf4d2dbcfcc73d981#file-loc-json)
![Comments](https://img.shields.io/endpoint?url=https://gist.github.com/RobinC94/7f38f27fb3b34d4bf4d2dbcfcc73d981#file-comments-json)

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

**DIgging** is released under the Apache 2.0 license.
