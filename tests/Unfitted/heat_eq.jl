using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = TProductModel(pmin,pmax,partition)
dp = pmax - pmin

cutgeo = cut(bgmodel,geo3)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ω_act = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)

order = 1
degree = 2*order+1
dΩ = Measure(Ω,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

stiffness(μ,u,v,dΩ,dΓ) = ∫( ∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
rhs(μ,v,dΩ,dΓ) = ∫( (γd/h)*v*fμ(μ) - (n_Γ⋅∇(v))*fμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ) = rhs(μ,v,dΩ,dΓ) - stiffness(μ,u,v,dΩ,dΓ)

reffe = ReferenceFE(lagrangian,Float64,order)

trians = (Ω,Γ)
test = AgFEMSpace(Ω_act,reffe,aggregates,conformity=:H1)
trial = ParamTrialFESpace(test,fμ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trians,trians)

index_map = FEOperatorIndexMap(trial,test)

vector_map = get_vector_index_map(test)
matrix_map = get_matrix_index_map(trial,test)

U = trial.space
V = test
sparsity = get_sparsity(U,V)
# psparsity = permute_sparsity(sparsity,U,V)
index_map_I = get_dof_index_map(V)
index_map_J = get_dof_index_map(U)
index_map_I_1d = get_tp_dof_index_map(V).indices_1d
index_map_J_1d = get_tp_dof_index_map(U).indices_1d

psparsity = permute_sparsity(sparsity.sparsity,index_map_I,index_map_J)
psparsities_1d = map(permute_sparsity,sparsity.sparsities_1d,is_1d,js_1d)
TProductSparsityPattern(psparsity,psparsities_1d)

STOP
# fesolver = LinearFESolver(LUSolver())

# tol = 1e-4
# energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
# state_reduction = Reduction(tol,energy;nparams=100,sketch=:sprn)
# rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

# R  = 0.5
# L  = 0.5*R
# p1 = Point(0.0,0.0)
# p2 = p1 + VectorValue(-L,L)

# geo1 = disk(R,x0=p1)
# geo2 = disk(R,x0=p2)
# geo3 = setdiff(geo1,geo2)

# t = 1.01
# pmin = p1-t*R
# pmax = p1+t*R

# n = 30
# partition = (n,n)
# bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
# dp = pmax - pmin

# cutgeo = cut(bgmodel,geo3)

# Ω_act = Triangulation(cutgeo,ACTIVE)
# Ω = Triangulation(cutgeo,PHYSICAL)

# order = 1
# reffe = ReferenceFE(lagrangian,Float64,order)
# Vstd = TestFESpace(Ω_act,reffe,conformity=:H1)

# strategy = AggregateAllCutCells()
# aggregates = aggregate(strategy,cutgeo)

# V = AgFEMSpace(Vstd,aggregates)
# U = TrialFESpace(V)

# degree = 2*order+1
# dΩ = Measure(Ω,degree)

# Γ = EmbeddedBoundary(cutgeo)
# n_Γ = get_normal_vector(Γ)
# dΓ = Measure(Γ,degree)

# const γd = 10.0    # Nitsche coefficient
# const h = dp[1]/n  # Mesh size according to the parameters of the background grid

# # add parameters
# pranges = fill([1,10],3)
# pspace = ParamSpace(pranges)

# f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
# f(μ) = x->f(x,μ)
# fμ(μ) = ParamFunction(f,μ)

# trians = (Ω,Γ)

# stiffness(μ,u,v,dΩ,dΓ) = ∫( ∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
# rhs(μ,v,dΩ,dΓ) = ∫( (γd/h)*v*fμ(μ) - (n_Γ⋅∇(v))*fμ(μ) )dΓ
# res(μ,u,v,dΩ,dΓ) = rhs(μ,v,dΩ,dΓ) - stiffness(μ,u,v,dΩ,dΓ)

# trial = ParamTrialFESpace(V,fμ)
# feop = LinearParamFEOperator(res,stiffness,pspace,trial,V,trians,trians)

# fesolver = LinearFESolver(LUSolver())

# tol = 1e-4
# energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
# state_reduction = Reduction(tol,energy;nparams=100,sketch=:sprn)
# rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)
