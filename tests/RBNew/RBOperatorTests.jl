# check on snapshots
using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using SparseArrays
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

θ = 1
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

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

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

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
rbinfo = RBInfo(dir;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=5)

rbsolver = RBSolver(rbinfo,fesolver)
snaps,comp = RB.collect_solutions(rbsolver,feop,uh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)

s = select_snapshots(snaps,1)
r = get_realization(s)
dtθ = dt*θ
y = zero_free_values(trial(r))
z = similar(y)
z .= 0.0
odeop = get_algebraic_operator(feop)
ode_cache = allocate_cache(odeop,r)
A_fe,b_fe = ODETools._allocate_matrix_and_vector(odeop,r,y,ode_cache)
ODETools._matrix_and_vector!(A_fe,b_fe,odeop,r,dtθ,y,ode_cache,z)

# case 1
red_trial = red_op.pop.trial
red_test = red_op.pop.test
op1 = GalerkinProjectionOperator(odeop,red_trial,red_test)

ode_cache = allocate_cache(op1,r)
ode_cache = update_cache!(ode_cache,op1,r)
A,b = allocate_fe_matrix_and_vector(op1,r,y,ode_cache)
A,b = RB.fe_matrix_and_vector!(A,b,op1,r,dtθ,y,ode_cache,z)

map(A,A_fe) do A,A_fe
  vA,vA_fe = get_values(A)[1],get_values(A_fe)[1]
  @test vA ≈ stack(nonzeros.(vA_fe.array))
end
@test sum(get_values(b)) ≈ stack(sum(get_values(b_fe)).array)

# case 2
op2 = red_op.pop

ode_cache = allocate_cache(op2,r)
ode_cache = update_cache!(ode_cache,op2,r)
A,b = ODETools._allocate_matrix_and_vector(op2,r,y,ode_cache)
A,b = fe_matrix_and_vector!(A,b,op2,r,dtθ,y,ode_cache,z)

@test sum(get_values(A))[indices_space,:] ≈ A_fe[indices_space,:]
@test sum(get_values(b)) ≈ b_fe
