using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
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
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)

########################## HEAT EQUATION ############################

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

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
info = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,st_mdeim=true)

rbsolver = RBSolver(info,fesolver)

snaps,comp = ode_solutions(rbsolver,feop,uh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)

son = select_snapshots(snaps,RB.online_params(info))
ron = get_realization(son)
xrb, = solve(rbsolver,red_op,ron)
son_rev = reverse_snapshots(son)
RB.space_time_error(son_rev,xrb,nothing)

info_space = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
rbsolver_space = RBSolver(info_space,fesolver)
red_op_space = reduced_operator(rbsolver_space,feop,snaps)
xrb_space, = solve(rbsolver_space,red_op_space,ron)
RB.space_time_error(son_rev,xrb_space,nothing)

xrb_loaded = load_solve(rbsolver)
xrb_space_loaded = load_solve(rbsolver_space)

#
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[3]

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = f(x,μ,t)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_h(x,t) = h(x,μ,t)
_h(t) = x->_h(x,t)

_b(t,v) = ∫(_f(t)*v)dΩ + ∫(_h(t)*v)dΓn
_a(t,du,v) = ∫(_a(t)*∇(v)⋅∇(du))dΩ
_m(t,dut,v) = ∫(v*dut)dΩ

_trial = TransientTrialFESpace(test,_g)
_feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
_u0 = interpolate_everywhere(x->0.0,_trial(0.0))

_sol = solve(fesolver,_feop,_u0,t0,tf)

dir = datadir("sol_CN")
createpvd(dir) do pvd
  for (uh,t) in _sol
    vtk = createvtk(Ω,dir*"sol_$(t).vtu",cellfields=["u"=>uh])
    pvd[t] = vtk
  end
end
