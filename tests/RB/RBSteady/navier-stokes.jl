using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using DrWatson
using Serialization

using GridapSolvers
import GridapSolvers: LinearSolvers, NonlinearSolvers, BlockSolvers

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ) = 1
a(μ) = x->a(x,μ)
aμ(μ) = ParamFunction(a,μ)

g(x,μ) = VectorValue(μ[1],0.0)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)
g0(x,μ) = VectorValue(0.0,0.0)
g0(μ) = x->g0(x,μ)
gμ0(μ) = ParamFunction(g0,μ)

const Re = 10.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

jac_lin(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res_lin(μ,(u,p),(v,q),dΩ) = jac_lin(μ,(u,p),(v,q),dΩ)

jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ
res_nlin(μ,(u,p),(v,q),dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["diri0","diri1"])
trial_u = ParamTrialFESpace(test_u,[gμ0,gμ])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = LinearParamFEOperator(res_lin,jac_lin,pspace,trial,test,trian_res,trian_jac)
feop_nlin = ParamFEOperator(res_nlin,jac_nlin,pspace,trial,test,trian_res,trian_jac)
feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)

fesolver = NonlinearFESolver(NonlinearSolvers.NewtonSolver(LUSolver();rtol=1e-10,maxiter=20))

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=30,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=10,nparams_jac=10)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,μon)

solver,op = rbsolver,rbop
r = μon

cache = solver.cache
y,paramcache = cache.fecache
x̂,rbcache = cache.rbcache
# linear + nonlinear cache
paramcache_lin,paramcache_nlin = paramcache
rbcache_lin,rbcache_nlin = rbcache

# linear cache
Âcache_lin,b̂cache_lin = rbcache_lin
op_lin = get_linear_operator(op)
op_nlin = get_nonlinear_operator(op)
Â_lin = jacobian!(Âcache_lin,op_lin,r,y,paramcache_lin)
b̂_lin = residual!(b̂cache_lin,op_lin,r,y,paramcache_lin)

# nonlinear cache
syscache_nlin = rbcache_nlin
trial = get_trial(op)(r)
cache = syscache_nlin,trial

nlop = RBNewtonOperator(op_nlin,paramcache_nlin,r,Â_lin,b̂_lin,cache)

using ReducedOrderModels.RBSteady

Â_lin,b̂_lin = RBSteady.get_linear_resjac(nlop)
syscache,trial = nlop.cache
Â_cache,b̂_cache = syscache

dx̂ = similar(x̂)
Â = jacobian!(Â_cache,nlop,y)
b̂ = residual!(b̂_cache,nlop,y)
# b̂ .+= Â_lin*x̂

TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(Â_lin),eltype(x̂))
similar(x̂,TS,axes(Â_lin,1))

similar(blocks(x̂)[1],Vector{Float64},Base.to_shape(blocks(axes(Â_lin,1))[1]))

x1 = param_getindex(x̂,1)
ax = axes(Â_lin,1)
similar(x1,Float64,ax)

for i in 1:2
  ai = blocks(x̂)[i]
  axi = map(ax -> blocks(ax)[i],ax)
  similar(ai,Vector{Float64},length.(axi))
end

A′ = map(blocks(A),blocks.(axes)) do a,ax
  similar(a,Array{T′,N},length.(ax))
end

# GRIDAP

μ1 = μ.params[1]
g_in′(x) = VectorValue(μ1[1],0.0)
g_0′(x) = VectorValue(0.0,0.0)

jac′((u,p),(du,dp),(v,q)) = ∫(a(μ1)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ + ∫(q*(∇⋅(du)))dΩ + jac_nlin(μ1,(u,p),(du,dp),(v,q),dΩ)
res′((u,p),(v,q)) = ∫(a(μ1)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + res_nlin(μ1,(u,p),(v,q),dΩ)

U = TrialFESpace(test_u,[g_0′,g_in′])
Y = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
X = MultiFieldFESpace([U,trial_p];style=BlockMultiFieldStyle())
feop′ = FEOperator(res′,jac′,X,Y)

uh,ph = solve(fesolver,feop′)
u = get_free_dof_values(uh)
p = get_free_dof_values(ph)


uh′ = zero(X)
# vh,cache = solve!(uh′,fesolver,feop′,nothing)
x′ = get_free_dof_values(uh′)
op′ = get_algebraic_operator(feop′)
# cache = solve!(x′,fesolver.nls,op′,nothing)
b′  = residual(op′,x′)
A′  = jacobian(op′,x′)
dx′ = allocate_in_domain(A′); fill!(dx′,zero(eltype(dx′)))
ss′ = symbolic_setup(fesolver.nls.ls,A′)
ns′ = numerical_setup(ss′,A′,x′)

r′ = norm(b′)
rmul!(b′,-1)
solve!(dx′,ns′,b′)
x′ .+= dx′

residual!(b′,op′,x′)
r′  = norm(b′)

jacobian!(A′,op′,x′)
numerical_setup!(ns′,A′,x′)
