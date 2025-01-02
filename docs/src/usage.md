# Usage

## Installation

ROM is a registered package. You can install it by running:

```julia
# Use ] to enter the Pkg REPL mode
pkg> add ROM
```

## Load package

Load the package normally with

```julia
using ROM
```

## Workflow 

The workflow of the package is analogous to that of [Gridap](https://github.com/gridap/Gridap.jl), a library for the grid-based approximation of PDEs. Comprehensive Gridap tutorials can be found [here](https://gridap.github.io/Tutorials/stable). We load Gridap with 

```julia
# Use ] to enter the Pkg REPL mode
pkg> add ROM
using Gridap
```

## Manufactured solution 

In this example we solve with a reduced order model a parameter dependent Poisson equation 

```julia
-ν*Δu = f in Ω
ν*∇u = h in Γ
```

where `Ω` is a sufficiently regular spatial domain, `ν` is a (positive) conductivity coefficient, `u` is the problem's unknown, `f` is a forcing term, and `h` a Neumann datum defined on the Neumann boundary `Γ`. In this problem, we consider `Ω = [0,1]^2` and `Γ` to be the right leg of the square. We consider the problem given by the following manufactured solution 

```julia
ν(μ) = exp(-sum(μ))
h(μ) = 1

u(x,μ) = μ⋅x 
f(x,μ) = -ν(μ)*Δ(u(x,μ))
```

Next, we parameterize `u` and `f` exclusively by `μ` in the following manner

```julia
uₚ(μ) = ParamFunction(x->u(x,μ),μ)
fₚ(μ) = ParamFunction(x->f(x,μ),μ)
```

A ParamFunction is a function that can be evaluated efficiently for any number of desired parameters. In a steady setting, it takes as argument a function (`u` and `f` in the cases above) and a parameter (in a transient setting, which we address later, an additional time variable must be included). 

## Geometry 

Define the geometry of the problem using 

```julia
Ω = (0,1,0,1)
partition = (10,10)
Ωₕ = CartesianDiscreteModel(Ω,partition)
```

## FE spaces 

Once the discrete geometry is introduced, we define a tuple of trial, test spaces (U,V) as

```julia
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(Ωₕ,reffe;dirichlet_tags=[1,3,5,6,7])
U = TrialParamFESpace(V,uₚ)
```

A `TrialParamFESpace` extends a traditional `TrialFESpace` in Gridap, as it allows to provide a `μ`-dependent Dirichlet datum. 

## Space of parameters 

We define a space of parameters, in this case `[1,10]^2`:

```julia 
D = ParamSpace((1,10,1,10))
```

A realization of `D` is a parameter, which in our case is a 2-dimensional vector. By default, a parameter is sampled according to a Halton sequence. Other sampling methods include uniform distributions 

```julia
ParamSpace((1,10,1,10),UniformSampling())
```

or normal distributions 

```julia
ParamSpace((1,10,1,10),NormalSampling())
```

## Numerical integration 

Before introducing the weak formulation of the problem, we define the quantities needed for the numerical integration 

```julia 
degree = 2*order
τₕ = Triangulation(Ωₕ)
Γₕ = BoundaryTriangulation(Ω;tags=[2,4,8])
dΩₕ = Measure(τₕ,degree)
dΓₕ = Measure(Γₕ,degree)
```

## Weak formulation 

Multiplying the Poisson equation `-ν*Δu = f` by a test function `v ∈ V` and integrating by parts yields the weak formulation of the problem, whose left- and right-hand (LHS & RHS) side are

```julia
a(μ,u,v,dΩₕ) = ∫(ν(μ)*∇(v)⋅∇(u))dΩₕ 
l(μ,v,dΩₕ,dΓₕ) = ∫(fₚ(μ)*v)dΩₕ + ∫(hₚ(μ)*v)dΓₕ
```

Note that, in contrast to a traditional Gridap code, the measures involved in the forms are arguments of the forms themselves. This is necessary in order to correctly run the hyper-reduction of the measures for the LHS & RHS, a step that occurs in the reduced order model

## Parametric FE problem 

At this point, we can build a FE operator representing the entire problem we aim to solve. 

```julia 
τₕ_l = FEDomains((Ωₕ,Γₕ))
τₕ_a = FEDomains((Ωₕ,))
feop = ParamLinearFEOperator(l,a,D,U,V,τₕ_l,τₕ_a)
```

A `FEDomains` is a structure collecting the triangulations relative to the measures in the LHS & RHS. With respect to a traditional FE operator in Gridap, a `ParamLinearFEOperator` provides the aforementioned `FEDomains` for the LHS & RHS, as well as the parametric domain `D`.

## FE solver 

We define the FE solver for our Poisson problem: 

``` julia
ls = LUSolver()
solver = LinearFESolver(ls)
```

## RB solver

Finally, we are ready to begin the ROM part. The first part consists in defining the problem's `RBSolver`, i.e. the reduced counterpart of a FE solver.

```julia 
tol = 1e-4
inner_prod(u,v) = ∫(∇(v)⋅∇(u))dΩₕ

reduction_sol = PODReduction(tol,inner_prod;nparams=20)
reduction_l = MDEIMReduction(tol;nparams=10)
reduction_a = MDEIMReduction(tol;nparams=10)
rbsolver = RBSolver(solver,reduction_sol,reduction_l,reduction_a)
```

A `RBSolver` contains the following information: 

* The FE solver 

* The reduction strategy for the solution. The output is a projection map representing a low-dimensional subspace, on which the HF solution is approximated. In the case above, we use a truncated POD with tolerance `tol` on a set of 20 snapshots. The output is orthogonal with respect to the form `inner_prod`, i.e. the `H^1_0` product

* The hyper-reduction strategy for the RHS. The output is a projection map representing a low-dimensional subspace for the RHS, equipped with a reduced integration domain obtained via MDEIM, i.e. the matrix-based empirical interpolation method. A total of 10 residual snapshots is used to compute the output

* The hyper-reduction strategy for the LHS, which is the matrix generalization of the RHS hyper-reduction

## Offline phase 

The offline phase is the part of the code where the approximations for the solution, LHS & RHS are computed. This phase is quite expensive to run, but on the upside it can just be run once, provided the outputs are saved to file. For this purpose, we load the well-known package [DrWatson][https://juliadynamics.github.io/DrWatson.jl/stable/]

```julia 
# Use ] to enter the Pkg REPL mode
pkg> add ROM
using DrWatson 
```

and define a saving directory 

```julia 
dir = datadir("poisson")
create_dir(dir) 
```

Next, we try loading the offline quantities; if the load fails, the offline phase must be run 

```julia 
try # try loading offline quantities
    rbop = load_operator(dir,feop)
catch # offline phase
    rbop = reduced_operator(rbsolver,feop)
    save(rbop,dir)
end
```

The load might fail, for e.g., if 

* It is the first time running the code

* A different directory was used to save the offline structures in previous runs 

* For developers, if the definition of one of the loaded types has changed since it was saved to file 

The offline structures are completely contained in the variable `rbop`, which is the reduced version of the FE operator `feop`. To understand better the meaning of this variable, we report the content of the function `reduced_operator`

```julia
# compute the solution snapshots 
fesnaps, = solution_snapshots(rbsolver,feop) 
# compute the reduced trial and test spaces 
Û,V̂ = reduced_fe_space(rbsolver,feop,fesnaps)
# compute the hyper-reduction for LHS & RHS
op = get_algebraic_operator(feop)
â,l̂ = reduced_weak_form(rbsolver,op,Û,V̂,fesnaps)

# fetch the reduced FEDomains
τₕ_l̂,τₕ_â = get_domains(l̂),get_domains(â)
# replace the original FEDomains with the new ones  
op′ = change_domains(op,τₕ_l̂,τₕ_â)
# definition of reduced operator 
rbop = GenericRBOperator(op′,Û,V̂,â,l̂)
```

## Online phase 

This step consists in computing the ROM approximation for any desired parameter. We consider, for e.g., 10 parameters distributed uniformly on `D`

```julia
μon = realization(D;nparams=10,rand=true)
```

and we solve the reduced problem 

```julia
x̂on,rbstats = solve(rbsolver,rbop,μon)
```

## Post processing 

In order to test the quality of the approximation `x̂on`, we can run the following post-processing code 

```julia
xon,festats = solve(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,xon,x̂on,festats,rbstats,μon)
println(perf)
```

In other words, we first compute the HF solution `xon` in the online parameters `μon`, and then we run the performance tester, which in particular returns

* the relative error ||xon - x̂on|| / ||xon||, averaged on the 10 parameters in `μon`. The norm is the one specified by `inner_prod`, so in our case the `H^1_0` product

* the speedup in terms of time and memory achieved with respect to the HF simulations. This is done by comparing the variables `rbstats` and `festats`, which contain the time (in seconds) and memory allocations (in Gb) of the two algorithms.