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
RB.space_time_error(son_rev,xrb)

info_space = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
rbsolver_space = RBSolver(info_space,fesolver)
red_op_space = reduced_operator(rbsolver_space,feop,snaps)
xrb_space, = solve(rbsolver_space,red_op_space,ron)
RB.space_time_error(son_rev,xrb_space)

xrb_loaded = load_solve(rbsolver)
xrb_space_loaded = load_solve(rbsolver_space)

#
# dummy test for online phase, no mdeim (θ == 1 !!)
son = select_snapshots(snaps,first(RB.online_params(info)))
x = get_values(son)
ron = get_realization(son)
odeop = get_algebraic_operator(feop.op)
ode_cache = allocate_cache(odeop,ron)
ode_cache = update_cache!(ode_cache,odeop,ron)
x0 = get_free_dof_values(zero(trial(ron)))
y0 = similar(x0)
y0 .= 0.0
nlop = ThetaMethodParamOperator(odeop,ron,dt*θ,x0,ode_cache,y0)
A = allocate_jacobian(nlop,x0)
jacobian!(A,nlop,x0,1)
Asnap = Snapshots(A,ron)
M = allocate_jacobian(nlop,x0)
jacobian!(M,nlop,x0,2)
Msnap = Snapshots(M,ron)
b = allocate_residual(nlop,x0)
residual!(b,nlop,x0)
bsnap = Snapshots(b,ron)

b_rb = compress(bsnap,red_test)
A_rb = compress(Asnap,red_trial(ron),red_test;combine=(x,y)->θ*x+(1-θ)*y)
M_rb = compress(Msnap,red_trial(ron),red_test;combine=(x,y)->θ*(x-y))
AM_rb = A_rb + M_rb

x_rb = AM_rb \ b_rb

x_rec = recast(red_trial(ron),x_rb)
s = Snapshots(x,ron)
srec = -Snapshots(x_rec,ron)
RB.space_time_error(s,srec)
