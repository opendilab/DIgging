# DIgging

<img src="./docs/figs/digging_banner.png" alt="icon"/>

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
[![Style](https://github.com/opendilab/digging/actions/workflows/style.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/style.yml?query=workflow%3A%22Style+And+Badge%22)
[![Test](https://github.com/opendilab/digging/actions/workflows/test.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/test.yml?query=workflow%3A%22Code+Test%22)
[![Docs](https://github.com/opendilab/digging/actions/workflows/doc.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/doc.yml?query=workflow%3A%22Docs+Deploy%22)
[![Package](https://github.com/opendilab/digging/actions/workflows/release.yml/badge.svg)](https://github.com/opendilab/DIgging/actions/workflows/release.yml?query=workflow%3A%22Package+Release%22)

[![PyPI](https://img.shields.io/pypi/v/digging)](https://pypi.org/project/digging/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/digging)](https://pypi.org/project/digging/)
[![codecov](https://img.shields.io/codecov/c/github/opendilab/digging)](https://app.codecov.io/gh/opendilab/digging)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/RobinC94/7f38f27fb3b34d4bf4d2dbcfcc73d981/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/RobinC94/7f38f27fb3b34d4bf4d2dbcfcc73d981/raw/comments.json)

![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/digging)](https://github.com/opendilan/digging/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/digging)](https://github.com/opendilab/digging/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/digging)
[![GitHub license](https://img.shields.io/github/license/opendilab/digging)](https://github.com/opendilab/digging/blob/master/LICENSE)

DIgging -- Decision Intelligence for digging better parameters in target function/environments.

## Introduction

**DIgging** is a heuristic searching and optimization platform with various Decision Intelligence methods such as Genetic Algorithm, Bayesian Optimization and Reinforcement Learning etc. It can be used to digging better candidates to handle combinatorial optimization problems and non-gradient search problems.

**DIgging** is a fundamental platform under [**OpenDILab**](http://opendilab.org/) and it uses [**DI-engine**](https://github.com/opendilab/DI-engine) to build RL searching pipelines.

[Documentation](https://opendilab.github.io/DIgging/main/index.html)

## Outlines

- [DIgging](#digging)
  - [Introduction](#introduction)
  - [Outlines](#outlines)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Digging Method Zoo](#digging-method-zoo)
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

**DIgging** defines core algorithm and searching methods as `Digger`. You can define a `Digger` with a searching `Space`,
and can be modified by a config dict.
**DIgging** provides two kinds of searching pipeline for a target function. Thet are listed as follow.

1. Interactive Procedure

It is done by calling `Digger`'s `propose` and `update_score` method, in which you can flexibly define the searching
procedures. You can call the `provide_best` method at any time to see the currently best candidate sample and its score.
Here's an simple example:

```python
def target_func(x):
    ...
	return score

space = YourSpace(shape=(...))
digger = YourDigger(config, space)

for i in range(max_iterations):
    samples = digger.propose(sample_num)
    scores = [target_func(x) for x in samples]
    digger.update_score(samples, scores)

print(digger.provide_best())
```

2. Functional Procedure

It is done by calling the `search` method of `Digger`, with target function provided as input. The digger will
automatically search the best samples of the target according to the config. Here's an example:

```python
def target_func(x):
    ...
    return score

space = YourSpace(shape=(...))
digger = YourDigger(config, space)

digger.search(target_func)

print(digger.provide_best())
```

3. Reinforcement Learning Procedure

When using a Reinforcement Learning `Digger`, users need to provide an RL `Policy` defined in **DI-engine** form,
and some other RL workers in **DI-engine** such as `Collector`, `Learner`, `ReplayBuffer` are supposed to be used
in the `Digger`. In the searching procedure, a target `Env` is used instead of a function. So we suggest to use
the `search` method to if the user is not familiar with the RL pipeline of **DI-engine**. Here's an example.

```python
def target_func(x):
    ...
    return score

rl_config = EasyDict(dict(...))
space = YourSearchSpace(shape=(...))
policy = YourPolicy(rl_config.policy, ...)
digger = RLDigger(rl_cfg, space, policy)

digger.search(target_func)

print(digger.provide_best())
```

## Digging Method Zoo

- Genetic Algorithm
- Bayesian Optimization
- RL

## License

**DIgging** is released under the Apache 2.0 license.
