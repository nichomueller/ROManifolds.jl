using Gridap
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)
order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

# weak formulation
a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

energy(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# solvers
fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = 1e-4
state_reduction = TransientReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=0)

test_dir = datadir(joinpath("heateq","elasticity_$(1e-4)"))
create_dir(test_dir)

# RB method

# fesnaps,festats = solution_snapshots(rbsolver,feop,uh0μ)
fesnaps = load_snapshots(test_dir)
# rbop = reduced_operator(rbsolver,feop,fesnaps)

# ronline = realization(feop;nparams=10)
# rbsnaps,rbstats = solve(rbsolver,rbop,ronline)

# fesnaps_test,_ = solution_snapshots(rbsolver,feop,uh0μ;r=ronline)

# results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

# println(results)

# TESTS

# rbop = reduced_operator(rbsolver,feop,fesnaps)
red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)

# red = RBSteady.get_state_reduction(rbsolver)
# basis = reduced_basis(red,feop,fesnaps)

# bs = basis.basis_space

# B = get_basis(bs)
# nfe = num_fe_dofs(bs)
# nrb = num_reduced_dofs(bs)
# x = rand(nfe)
# xrb = rand(nrb)
# project(bs,x) ≈ B'*x
# inv_project(bs,xrb) ≈ B*xrb

# galerkin_projection(bs,bs)
# empirical_interpolation(bs)

# using LinearAlgebra
# A = rand(nfe,nfe)
# U,S,V = svd(A)
# r = rank(A)
# U = U[:,1:r]

# X = assemble_matrix(feop,energy)
# bbs = union(bs,PODBasis(U),X)

using Gridap.FESpaces
op = get_algebraic_operator(feop)
pop = TransientPGOperator(op,red_trial,red_test)

# r = realization(feop)
# Û = red_trial(r)
# get_vector_type(Û)

# cache = zero_free_values(Û)

# jacs = jacobian_snapshots(rbsolver,pop,fesnaps)
ress = residual_snapshots(rbsolver,pop,fesnaps)

# red_jac = reduced_jacobian(rbsolver.jacobian_reduction,pop,jacs)
red_res = reduced_residual(rbsolver.residual_reduction,pop,ress)

reduction = get_reduction(rbsolver.residual_reduction)
basis = projection(reduction,ress[2])
proj_basis = project(red_test,basis)
indices,interp = empirical_interpolation(basis)
factor = lu(interp)
domain = integration_domain(indices)

rbop = reduced_operator(rbsolver,feop,fesnaps)

r = realization(feop;nparams=10)
# rbsnaps,rbstats = solve(rbsolver,rbop,r)
using Gridap.ODEs
using Gridap.FESpaces
using Gridap.Algebra

op = rbop
x̂ = zero_free_values(get_trial(op)(r))
y = zero_free_values(get_fe_trial(op)(r))
RBSteady.init_online_cache!(rbsolver,op,r,y)

# solve!(x̂,rbsolver,op,r)
cache = rbsolver.cache
y,odecache = cache.fecache
rbcache = cache.rbcache
# solve!((x̂,),fesolver,op,r,(y,),(odecache,rbcache))
odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache
Â,b̂ = rbcache
us = (y,y)
ws = (1,1/(dt*θ))
# stageop = LinearParamStageOperator(op,odeopcache,r,us,ws,(A,Â),(b,b̂),reuse,sysslvrcache)
residual!((b,b̂),op,r,us,odeopcache)
jacobian!((A,Â),op,r,us,ws,odeopcache)



fe_sb = fe_residual!(b,op,r,us,odeopcache)
inv_project!(b̂,op.rhs,fe_sb)

coeff,result = b̂[1][2],b̂[2]
b = fe_sb[2]
hyp = op.rhs[2]

interp = RBSteady.get_interpolation(hyp)
ldiv!(coeff,interp,vec(b))
muladd!(result,hyp,coeff)

lhs1 = RBSteady.allocate_coefficient(op.lhs[1][1],r)
lhs2 = RBSteady.allocate_hyper_reduction(op.lhs[1][1],r)

rhs1 = RBSteady.allocate_coefficient(op.rhs[1],r)
rhs2 = RBSteady.allocate_hyper_reduction(op.rhs[1],r)

x = inv_project(get_trial(op)(r),x̂)
