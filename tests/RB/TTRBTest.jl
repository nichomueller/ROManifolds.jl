using Gridap
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (10,10)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

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

induced_norm(du,v) = ∫(v*u)dΩ + ∫(∇(v)⋅∇(du))dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ;tt_format=true)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
println(RB.speedup(results))

save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.Helpers
using BlockArrays

using Mabla.FEM
using Mabla.RB

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)

test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])

# reffe1 = ReferenceFE(QUAD,lagrangian,Float64,order)
# reffe2 = ReferenceFE(QUAD,lagrangian,VectorValue{2,Float64},order)

# _reffe1 = ReferenceFE(QUAD,FEM.tplagrangian,Float64,order)

# basis,reffe_args,reffe_kwargs = reffe
# cell_reffe = ReferenceFE(model,basis,reffe_args...;reffe_kwargs...)

# _reffe = ReferenceFE(FEM.tplagrangian,Float64,order)
# _basis,_reffe_args,_reffe_kwargs = _reffe
# _cell_reffe = ReferenceFE(model,_basis,_reffe_args...;_reffe_kwargs...)
# conf = Conformity(testitem(_cell_reffe),:H1)

# cell_fe = CellFE(model,_cell_reffe,conf)
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

stiff(du,v) = ∫(∇(du)⋅∇(v))dΩ
U = TrialFESpace(test,x->sum(x))
V = test
assemble_matrix(stiff,U,V)


a = SparseMatrixAssembler(U,V)
v = get_fe_basis(V)
u = get_trial_fe_basis(U)

∇(u)⋅∇(v)

function Arrays.return_cache(b::LagrangianDofBasis,field)
  error("stop here")
  cf = return_cache(field,b.nodes)
  vals = evaluate!(cf,field,b.nodes)
  ndofs = length(b.dof_to_node)
  r = _lagr_dof_cache(vals,ndofs)
  c = CachedArray(r)
  (c, cf)
end
