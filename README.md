# ROManifolds

This package provides a set of tools for the solution of parameterized partial differential equations (PDEs) with reduced order models (ROMs). The presence of parameters severely impacts the feasibility of running high-fidelity (HF) codes such as the finite element (FE) method, because typically the solution is required for many different values of the parameters. ROMs create surrogate models that approximate the solution manifold on a lower-dimensional manifold. These surrogates provide accurate solutions in a much shorter time and with much fewer computational resources. The library is developed in close collaboration with [Gridap.jl](https://github.com/gridap/Gridap.jl).

| **Documentation** |
|:------------ |
| [![docdev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichomueller.github.io/ROManifolds.jl/dev/) |
| **Citation** |
| [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2022.111162-blue)](https://github.com/nichomueller/rb_julia) |
|**Build Status** |
| [![CI](https://github.com/nichomueller/ROManifolds.jl/workflows/CI/badge.svg)](https://github.com:nichomueller/ROManifolds.jl/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/github.com:nichomueller/ROManifolds.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/github.com:nichomueller/ROManifolds.jl) |

## Installation

```julia
# Type ] to enter package mode
pkg> add ROManifolds
```

## Examples

Before running the following examples, it is necessary to import from file some geometries saved to `.msh` file. They can be found in the [assets](https://nichomueller.github.io/ROManifolds.jl/docs/assets) directory of this repo, in the zipped file named `models.tar.gz`. The geometries must be unzipped and moved to a directory where the numerical experiments are ran. This directory should be placed inside the `data` directory of the `Julia` project which is being used to run these experiments. To find this directory, first add the package `DrWatson` with 

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

In the following numerical examples, we provide a plot of the convergence errors for a series of tolerances (which determine the accuracy of the method), and a plot solution obtained with a fixed tolerance of `1e-5`.

### Test 1 

Solve a steady elasticity problem with a proper orthogonal decomposition algorithm. 

```julia
julia> include("examples/SteadyElasticityPOD.jl")
```
Solution             |  Convergence
:-------------------------:|:-------------------------:
![Example 1 solution](docs/src/assets/results/elasticity_pod/plot/rbsol.png)  |  ![Example 1 convergence](docs/src/assets/results/results/elasticity_pod/results/convergence.pdf)

### Test 2

Do the same, but with a tensor-train decomposition approach.

```julia
julia> include("examples/SteadyElasticityTTSVD.jl")
```
Solution             |  Convergence
:-------------------------:|:-------------------------:
<img src="docs/src/assets/results/elasticity_ttsvd/plot/rbsol.png" alt="drawing" height="100" width="100"/>  |  <img src="docs/src/assets/results/elasticity_ttsvd/results/convergence.pdf" alt="drawing" height="100" width="100"/>

### Test 3

Solve a steady Stokes equation with a proper orthogonal decomposition algorithm.

![Example 3](examples/example3.png)

```julia
julia> include("examples/SteadyStokesPOD.jl")
```

Solution-velocity          |  Solution-pressure        |  Convergence
:-------------------------:|:-------------------------:|:-------------------------:
![Example 3 velocity](docs/src/assets/results/stokes_pod/plot/rbvel.png) | ![Example 3 pressure](docs/src/assets/results/stokes_pod/plot/rbpress.png)  |  ![Example 3 convergence](docs/src/assets/results/stokes_pod/results/results/convergence.pdf)

### Test 4 

Moving to transient applications, we first solve a heat equation with a space-time RB method.

```julia
julia> include("examples/HeatEquationSTRB.jl")
```

Solution             |  Convergence
:-------------------------:|:-------------------------:
![Example 2 solution](docs/src/assets/results/heateq_pod/plot/rbsol.gif)  |  ![Example 2 convergence](docs/src/assets/results/heateq_pod/results/results/convergence.pdf)

### Test 5

Lastly, we solve a Navier-Stokes equation with a space-time RB method.

```julia
julia> include("examples/NStokesTransientSTRB.jl")
```

<!-- |![Example 3 velocity](docs/src/assets/results/transient_nstokes_pod/plot/rbvel.gif) ![Example 3 pressure](docs/src/assets/results/transient_nstokes_pod/plot/rbpress.gif) ![Example 3 convergence](docs/src/assets/results/transient_nstokes_pod/results/convergence.pdf)|  -->
