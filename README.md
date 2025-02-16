# ROM

This package provides a set of tools for the solution of parameterized partial differential equations (PDEs) with reduced order models (ROMs). The presence of parameters severely impacts the feasibility of running high-fidelity (HF) codes such as the finite element (FE) method, because typically the solution is required for many different values of the parameters. ROMs create surrogate models that approximate the solution manifold on a lower-dimensional manifold. These surrogates provide accurate solutions in a much shorter time and with much fewer computational resources. The library is developed in close collaboration with [Gridap.jl](https://github.com/gridap/Gridap.jl).

| **Documentation** |
|:------------ |
| [![docdev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com:nichomueller/ROM.jl/dev) |
| **Citation** |
| [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2022.111162-blue)](https://github.com/nichomueller/rb_julia) |
|**Build Status** |
| [![CI](https://github.com:nichomueller/ROM.jl/workflows/CI/badge.svg)](https://github.com:nichomueller/ROM.jl/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/github.com:nichomueller/ROM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/github.com:nichomueller/ROM.jl) |

## Installation

```julia
# Type ] to enter package mode
pkg> add ROM
```

## Examples

### Sub-triangulation examples

Use a test geometry, e.g., `47076.stl` (Chichen Itza)
```julia
julia> include("examples/SubTriangulation.jl")
julia> filename = "test/data/47076.stl"
julia> SubTriangulation.main(filename,n=50,output="example1")
```
![Example 1](examples/example1.png)

Download a geometry directly from [Thingi10k](https://ten-thousand-models.appspot.com/), e.g, [37384](https://ten-thousand-models.appspot.com/detail.html?file_id=37384). Please check whether the geometry is *solid* and *manifold* in Thingi10k metadata.
```julia
julia> include("examples/SubTriangulation.jl")
julia> filename = SubTriangulation.download(37384)
julia> SubTriangulation.main(filename,n=50,output="example2")
```
![Example 2](examples/example2.png)

### Finite Elements examples

Solve a **Poisson** equation on a test geometry, e.g., `293137.stl` (Low-Poly Bunny)
 ```julia
julia> include("examples/Poisson.jl")
julia> filename = "test/data/293137.stl"
julia> Poisson.main(filename,n=20,output="example3")
```

![Example 3](examples/example3.png)

Solve a **Linear Elasticity** problem on a test geometry, e.g., `550964.stl` (Eiffel Tower in a 5 degree slope)
 ```julia
julia> include("examples/LinearElasticity.jl")
julia> filename = "test/data/550964.stl"
julia> LinearElasticity.main(filename,n=50,force=(tand(5),0,-1),output="example4")
```

![Example 4](examples/example4.png)

Solve an **Incompressible Flow** problem on a test geometry, e.g., `47076.stl` (Chichen Itza)
 ```julia
julia> # ENV["ENABLE_MKL"] = "" ## Uncomment if GridapPardiso.jl requirements are fulfilled
julia> include("examples/Stokes.jl")
julia> filename = "test/data/47076.stl"
julia> Stokes.main(filename,n=10,output="example5")
```

![Example 5](examples/example5.png)