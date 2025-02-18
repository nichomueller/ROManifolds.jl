# ROM

This package provides a set of tools for the solution of parameterized partial differential equations (PDEs) with reduced order models (ROMs). The presence of parameters severely impacts the feasibility of running high-fidelity (HF) codes such as the finite element (FE) method, because typically the solution is required for many different values of the parameters. ROMs create surrogate models that approximate the solution manifold on a lower-dimensional manifold. These surrogates provide accurate solutions in a much shorter time and with much fewer computational resources. The library is developed in close collaboration with [Gridap.jl](https://github.com/gridap/Gridap.jl).

| **Documentation** |
|:------------ |
| [![docdev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichomueller.github.io/ROM.jl/dev/) |
| **Citation** |
| [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2022.111162-blue)](https://github.com/nichomueller/rb_julia) |
|**Build Status** |
| [![CI](https://github.com/nichomueller/ROM.jl/workflows/CI/badge.svg)](https://github.com:nichomueller/ROM.jl/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/github.com:nichomueller/ROM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/github.com:nichomueller/ROM.jl) |

## Installation

```julia
# Type ] to enter package mode
pkg> add ROM
```

## Examples

Before running the following examples, it is necessary to import from file some geometries saved to `.msh` file. They can be found in the [assets](https://nichomueller.github.io/ROM.jl/docs/assets) directory of this repo, in the zipped file named `models.tar.gz`. The geometries must be unzipped and moved to a directory where the numerical experiments are ran. This directory should be placed inside the `data` directory of the `Julia` project which is being used to run these experiments. To find this directory, first add the package `DrWatson` with 

```julia
# Type ] to enter package mode
pkg> add DrWatson
```

and call

```julia
julia> dir = datadir()
```

Now we can unzip the compressed folder in `dir` with 

```julia
# Type ; to enter shell mode
shell> tar -xvzf models.tar.gz -C dir
```

### Test 1 

Solve a steady elasticity problem with a proper orthogonal decomposition algorithm

```julia
julia> include("examples/SteadyElasticityPOD.jl")
```
![Example 1](examples/example1.png)

### Test 2

Do the same, but with a tensor-train decomposition approach

```julia
julia> include("examples/SteadyElasticityTTSVD.jl")
```
![Example 2](examples/example2.png)

### Test 3

Solve a steady Stokes equation with a proper orthogonal decomposition algorithm

![Example 3](examples/example3.png)

```julia
julia> include("examples/SteadyStokesPOD.jl")
```

### Test 4 

![Example 4](examples/example4.png)

```julia
julia> include("examples/SteadyStokesTTSVD.jl")
```

### Test 5 

Moving to transient applications, we first solve a heat equation with a space-time RB method

```julia
julia> include("examples/HeatEquationSTRB.jl")
```

![Example 5](examples/example5.png)

### Test 6

Lastly, we solve a Navier-Stokes equation with a space-time RB method

```julia
julia> include("examples/NStokesTransientSTRB.jl")
```

![Example 6](examples/example6.png)