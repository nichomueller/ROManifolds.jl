using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField

θ = 0.5
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

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

res(μ,t,u,v) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ
induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

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
feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

x0 = zero_free_values(trial(r))
op = get_algebraic_operator(feop)
ode_cache = allocate_cache(op,r)
A,b = ODETools._allocate_matrix_and_vector(op,r,x0,ode_cache)

tt_test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6],vector_type=TTVector{1,Float64})
tt_trial = TransientTrialParamFESpace(tt_test,gμt)
tt_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,tt_trial,tt_test)
tt_uh0μ(μ) = interpolate_everywhere(u0μ(μ),tt_trial(μ,t0))

tt_x0 = zero_free_values(tt_trial(r))
tt_op = get_algebraic_operator(tt_feop)
tt_ode_cache = allocate_cache(tt_op,r)
tt_A,tt_b = ODETools._allocate_matrix_and_vector(tt_op,r,tt_x0,tt_ode_cache)
