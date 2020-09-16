# MultiOptim

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://robertfeldt.github.io/MultiOptim.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://robertfeldt.github.io/MultiOptim.jl/dev)
[![Build Status](https://github.com/robertfeldt/MultiOptim.jl/workflows/CI/badge.svg)](https://github.com/robertfeldt/MultiOptim.jl/actions)
[![Coverage](https://codecov.io/gh/robertfeldt/MultiOptim.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/robertfeldt/MultiOptim.jl)

`MultiOptim` is a global optimization package for Julia (http://julialang.org/). It supports both multi- and single-objective optimization problems and is focused on (meta-)heuristic/stochastic algorithms (DE, NES etc) that do NOT require the function being optimized to be differentiable. This is in contrast to more traditional, deterministic algorithms that are often based on gradients/differentiability. It also supports parallel evaluation to speed up optimization for functions that are slow to evaluate.

`MultiOptim` is the successor to the [BlackBoxOptim.jl](https://robertfeldt.github.io/BlackBoxOptim.jl) and tries to keep what was good while fixing the bad parts. In particular, it supports multi-objective optimization and can also optimize functions that require values of different types (i.e. not only vectors for floats).

For now, `MultiOptim` is not ready for prime time since we are exploring design options. Please come back soon and we will be up and running. :)