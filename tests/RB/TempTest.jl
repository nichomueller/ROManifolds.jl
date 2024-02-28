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
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[3]

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
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

b(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
a(μ,t,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
m(μ,t,dut,v,dΩ) = ∫(v*dut)dΩ

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
info = RBInfo(dir;nsnaps_state=5,nsnaps_test=5)

snaps,comp = RB.fe_solutions(info,fesolver,feop,uh0μ)
bs,bt = reduced_basis(info,feop,snaps)

x1 = ParamArray(RB.tensor_getindex(snaps,:,:,1))
r1 = snaps.realization[1,:]
trial1 = trial(r1)
x1h = FEFunction(trial1,x1)

create_dir(dir)
for (k,(μ,t)) in enumerate(r1)
  x1hk = FEM._getindex(x1h,k)
  file = joinpath(dir,"solution_$t"*".vtu")
  writevtk(Ω,file,cellfields=["u"=>x1hk])
end

########

rbsolver = RB.RBSolver(info,fesolver)

rb_test = RBSpace(test,bs,bt)
rb_trial = RBSpace(trial,bs,bt)

# op = RBOperator(get_algebraic_operator(feop),rb_trial,rb_test)
# nparams = RB.num_mdeim_params(rbsolver.info)
# r = realization(op.feop;nparams)
# nlop = RB.get_method_operator(rbsolver,op,r)
# x = nlop.u0
# bb = Algebra.residual(nlop,x)
# AA = Algebra.jacobian(nlop,x)

trian_res = [Ω]
trian_jac = ([Ω],[Ω])
feop_trian = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test,trian_res,trian_jac)

op = RBOperator(get_algebraic_operator(feop_trian),rb_trial,rb_test)
nparams = RB.num_mdeim_params(rbsolver.info)
r = realization(op.feop;nparams)
nlop = RB.get_method_operator(rbsolver,op,r)
x = nlop.u0
bb = Algebra.residual(nlop,x)
AA = Algebra.jacobian(nlop,x)

snapb = Snapshots(bb,r)
snapA = map(AA) do AA
  Snapshots(AA,r)
end

acv = RB.reduced_vector_form(rbsolver,op,snapb)
acm = RB.reduced_matrix_form(rbsolver,op,snapA)

s = snapA[1][Ω]
basis_spaceA,basis_timeA = RB.compute_bases(s)

red_op = RB.reduced_operator(rbsolver,op)
