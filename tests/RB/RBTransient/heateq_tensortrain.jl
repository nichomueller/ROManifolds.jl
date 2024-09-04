using Gridap
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.Utils
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
n = 5
domain = (0,1,0,1,0,1)
partition = (n,n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"neumann",[21,22])
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,24,25,26])

order = 1#2
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

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = fill(1e-4,4)
state_reduction = TTSVDReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_test=5,nparams_res=30,nparams_jac=20)
test_dir = datadir(joinpath("heateq","test_tt_$(1e-4)"))
create_dir(test_dir)

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats,cache = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(results)

save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

ranks = map(c -> size(c,3),RBSteady.get_cores(RBSteady.get_basis(rbop.op.test)))
_state_reduction = TTSVDReduction(ranks,energy;nparams=50)
_rbsolver = RBSolver(fesolver,state_reduction;nparams_test=5,nparams_res=30,nparams_jac=20)

_rbop = reduced_operator(_rbsolver,feop,fesnaps)
_rbsnaps,_rbstats,_cache = solve(_rbsolver,_rbop,fesnaps)
_results = rb_results(_rbsolver,_rbop,fesnaps,_rbsnaps,festats,_rbstats)

println(_results)
