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
using Gridap.TensorValues
using BlockArrays
using DrWatson
using Kronecker
using Mabla.FEM
using Mabla.RB

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.05

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (5,5)
model = TProductModel(domain,partition)

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

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)

# red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# odeop = get_algebraic_operator(feop)
# pop = PODOperator(odeop,red_trial,red_test)
# red_lhs,red_rhs = reduced_jacobian_residual(rbsolver,pop,fesnaps)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))

# OLD CODE
_model = CartesianDiscreteModel(domain,partition)

_labels = get_face_labeling(_model)
add_tag_from_tags!(_labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(_labels,"neumann",[7])

_Ω = Triangulation(_model)
_dΩ = Measure(_Ω,degree)
_Γn = BoundaryTriangulation(_model,tags=["neumann"])
_dΓn = Measure(_Γn,degree)

_trian_res = (_Ω,_Γn)
_trian_stiffness = (_Ω,)
_trian_mass = (_Ω,)

induced_norm(du,v) = ∫(v*du)_dΩ + ∫(∇(v)⋅∇(du))_dΩ

_test = TestFESpace(_model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial = TransientTrialParamFESpace(_test,gμt)
_feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  _trial,_test,_trian_res,_trian_stiffness,_trian_mass)
_uh0μ(μ) = interpolate_everywhere(u0μ(μ),_trial(μ,t0))

_fesnaps = RB.TransientOldTTSnapshots(fesnaps.values,fesnaps.realization)
# _rbop = reduced_operator(rbsolver,_feop,_fesnaps)
# _red_trial,_red_test = reduced_fe_space(rbsolver,_feop,_fesnaps)
_soff = select_snapshots(_fesnaps,RB.offline_params(rbsolver))
_red_trial,_red_test = reduced_fe_space(rbsolver,_feop,_soff)

_pop = PODOperator(get_algebraic_operator(_feop),_red_trial,_red_test)
smdeim = select_snapshots(_fesnaps,RB.mdeim_params(rbsolver))
_jac,_res = jacobian_and_residual(rbsolver,_pop,smdeim)
i0 = FEM.vectorize_index_map(FEM.get_free_dof_permutation(test))
b1 = ParamArray(map(a->TTArray(a,i0),_res.values[1].values))
b2 = ParamArray(map(a->TTArray(a,i0),_res.values[2].values))
vr1 = RB.OldSnapshots(b1,_res.values[1].realization)
vr2 = RB.OldSnapshots(b2,_res.values[2].realization)
_res = Contribution((vr1,vr2),_res.trians)
A1 = ParamArray(map(a->TTArray(a,i0),_jac[1].values[1].values))
A2 = ParamArray(map(a->TTArray(a,i0),_jac[2].values[1].values))
vj1 = RB.OldSnapshots(A1,_jac[1].values[1].realization)
vj2 = RB.OldSnapshots(A2,_jac[2].values[1].realization)
_jac = (Contribution((vj1,),_jac[1].trians),Contribution((vj2,),_jac[2].trians))
red_jac = RB.reduced_jacobian(rbsolver,_pop,_jac)
red_res = RB.reduced_residual(rbsolver,_pop,_res)
