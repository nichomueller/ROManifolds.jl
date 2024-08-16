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

induced_norm(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# solvers
fesolver = ThetaMethod(LUSolver(),dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_res=20,nsnaps_jac=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# RB method
# we can load & solve directly, if the offline structures have been previously saved to file
try
  results = load_solve(rbsolver,feop,test_dir)
catch
  @warn "Loading offline structures failed: running offline phase"
  fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

  save(test_dir,fesnaps)
  save(test_dir,rbop)
  save(test_dir,results)
end

# post process
println(compute_error(results))
println(compute_speedup(results))

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

# using Gridap.FESpaces
# red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# op = get_algebraic_operator(feop)
# pop = TransientPGOperator(op,red_trial,red_test)
# smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
# jjac,rres = jacobian_and_residual(rbsolver,pop,smdeim)

# isa(jjac[1][1],
#   RBTransient.TransientSnapshotsAtIndices{T,N,L,D,I,R,
#   <:RBTransient.TransientStandardSparseSnapshots{T,N,L,D,I,R,<:MatrixOfSparseMatricesCSC},B,C
#   } where {T,N,L,D,I,R,B,C})
# isa(jjac[1][1],TransientSparseSnapshots)

# const _TransientSparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC,B,C} = Union{
#   RBTransient.TransientStandardSparseSnapshots{T,N,L,D,I,R,A},
#   RBTransient.TransientSnapshotsAtIndices{T,N,L,D,I,R,
#   RBTransient.TransientStandardSparseSnapshots{T,N,L,D,I,R,A},B,C}
# }
# isa(jjac[1][1],_TransientSparseSnapshots)

# S1{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = RBTransient.TransientStandardSparseSnapshots{T,N,L,D,I,R,A}
# SSSS2{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = Union{
#   S1{T,N,L,D,I,R,A},RBTransient.TransientSnapshotsAtIndices{T,N,L,D,I,R,<:S1{T,N,L,D,I,R,A}}
# }

# const S1{T,N,L,D,I,R} = RBTransient.TransientStandardSparseSnapshots{T,N,L,D,I,R,<:MatrixOfSparseMatricesCSC}
# # const S2{T,N,L,D,I,R}
# isa(jjac[1][1],SSSS2)
