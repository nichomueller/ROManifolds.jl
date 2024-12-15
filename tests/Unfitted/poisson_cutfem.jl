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
dp = pmax - pmin

n = 30
partition = (n,n)
bgmodel = TProductModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

order = 1
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

nΓ = get_normal_vector(Γ)
nΓg = get_normal_vector(Γg)

const γd = 10.0
const γg = 0.1
const h = dp[1]/n

ν(x,μ) = 1+exp(-x[1]/sum(μ))
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g0(x,μ) = 0
g0(μ) = x->g0(x,μ)
g0μ(μ) = ParamFunction(g0,μ)

a(μ,u,v,dΩ,dΩ_out,dΓ,dΓg) = ( ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( ∇(v)⋅∇(u) )dΩ_out
  + ∫( (γd/h)*v*u  - νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
  + ∫( (γg*h)*jump(nΓg⋅∇(v))*jump(nΓg⋅∇(u)) ) * dΓg
  )

b(μ,u,v,dΩ,dΓ) = ∫( (γd/h)*v*fμ(μ) - νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*u - (nΓ⋅∇(v))*u ) * dΓ

reffe = ReferenceFE(lagrangian,Float64,order)

domains = FEDomains((Ω,Γ),(Ω,Ω_out,Γ,Γg))
test = TProductFESpace(Ωbg,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,g0μ)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10,random=true)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

# xrb = inv_project(rbop.test.subspace,x̂[1])
# Son = reshape(collect(x),:,10)

# u1 = flatten_snapshots(x)[:,1]
# r1 = get_realization(x)[1]
# U1 = param_getindex(trial(r1),1)
# uh = FEFunction(U1,u1)
# writevtk(Ω,datadir("plts/sol"),cellfields=["uh"=>uh])


# xrb = Snapshots(inv_project(rbop.trial(μon),x̂),get_dof_map(feop),μon)
# urb1 = flatten_snapshots(xrb)[:,1]
# uhrb1 = FEFunction(U1,urb1)
# writevtk(Ω,datadir("plts/sol_approx"),cellfields=["uh"=>uhrb1])

# using Gridap.CellData
# using SparseArrays
# using Gridap.Algebra
# using Gridap.FESpaces
# using Gridap.Geometry

# using ReducedOrderModels.Utils
# using ReducedOrderModels.DofMaps
# using ReducedOrderModels.RBSteady

# using BlockArrays
# using LinearAlgebra

# using Gridap.Algebra

# cores = rbop.test.subspace.cores
# Φ = get_basis(rbop.test.subspace)
# S = vec(fesnaps[:,:,1])
# X = assemble_matrix(feop,energy)
# RBSteady.check_orthogonality(cores,X)
# Xglobal = kron(X)
# E = S - Φ*Φ'*Xglobal*S

# S = collect(fesnaps)
# cc = reduction(state_reduction,S,X)
# @assert cc[1] ≈ cores[1]
# @assert cc[2] ≈ cores[2]

# op = get_algebraic_operator(feop)
# us = get_values(fesnaps) |> similar
# fill!(us,zero(eltype2(us)))
# rs = get_realization(fesnaps)

# jacs = jacobian_snapshots(rbsolver,op,fesnaps)
# ress = residual_snapshots(rbsolver,op,fesnaps)

# basis = projection(state_reduction,ress[1])
# # proj_basis = galerkin_projection(rbop.test.subspace,basis)
# proj_basis = galerkin_projection(rbop.test.subspace.cores,basis.cores)





# op = rbop
# r = μon
# xfe = zero_free_values(get_fe_trial(op)(r))
# x̂rb = zero_free_values(get_trial(op)(r))

# rbcache = allocate_rbcache(op,r,xfe)
# Â = jacobian(op,r,xfe,rbcache)
# b̂ = residual(op,r,xfe,rbcache)

# Â1 = param_getindex(Â,1)
# b̂1 = param_getindex(b̂,1)

# sjac = select_snapshots(x,1)
# us_jac = get_values(sjac) |> similar
# fill!(us_jac,zero(eltype2(us_jac)))
# r_jac = get_realization(sjac)
# oop = get_algebraic_operator(set_domains(feop))
# J1 = param_getindex(jacobian(oop,r_jac,us_jac),1)
# R1 = param_getindex(residual(oop,r_jac,us_jac),1)

# Φ = get_basis(op.test)
# Jrb = Φ'*J1*Φ
# err_Jac = Jrb - Â1

# Rrb = Φ'*R1
# err_Res = Rrb - b̂1


# # trian by trian

# op = get_algebraic_operator(feop)
# jacs = jacobian(op,r_jac,us_jac)
# ress = residual(op,r_jac,us_jac)

# res1 = Φ'*ress[1][1]
# res2 = Φ'*ress[2][1]

# B = rbcache.b.fe_quantity
# paramcache = rbcache.paramcache
# feb = fe_residual!(B,rbop,r,xfe,paramcache)

# coeff,hypred = rbcache.b.coeff,rbcache.b.hypred
# fill!(hypred,zero(eltype(hypred)))
# res1_rb = copy(hypred)
# res2_rb = copy(hypred)

# inv_project!((coeff[1],res1_rb),rbop.rhs[1],feb[1])
# inv_project!((coeff[2],res2_rb),rbop.rhs[2],feb[2])

# res1 - res1_rb[1]
# res2 - res2_rb[1]

# inds1 = rbop.rhs[1].domain
# feb[1][1] ≈ ress[1][1][inds1]

# x1 = x.data[1]
# x1 - Φ*Φ'*x1
