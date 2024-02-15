using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays
using DrWatson

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[1]

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1] + μ[2]*sin(2*π*t/μ[3])
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = sin(π*t/μ[3])
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = 0.0
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

res(μ,t,u,v,dΩ) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir("distr_toy_heateq")
info = RBInfo(dir;nsnaps_state=10,nsnaps_mdeim=5,nsnaps_test=5,save_structures=false)

rbsolver = RBSolver(info,fesolver)

# snaps,comp = fe_solutions(rbsolver,feop,uh0μ)

snaps = with_debug() do distribute
  load_distributed_snapshots(distribute,info)
end

# red_op = reduced_operator(rbsolver,feop,snaps)
red_trial,red_test = reduced_fe_space(info,feop,snaps)

sk = select_snapshots(snaps,15)
pk = get_values(sk)

pk_rb = compress(red_test,sk)
pk_rec = recast(red_test,pk_rb)

norm(pk_rec - pk) / norm(pk)

odeop = get_algebraic_operator(feop)
pop = GalerkinProjectionOperator(odeop,red_trial,red_test)
# red_lhs,red_rhs = reduced_matrix_vector_form(rbsolver,pop,snaps)

θ == 0.0 ? dtθ = dt : dtθ = dt*θ
smdeim = select_snapshots(snaps,RB.mdeim_params(info))
x = get_values(smdeim)
r = get_realization(smdeim)

y = similar(x)
y .= 0.0
ode_cache = allocate_cache(pop,r)
A,b = allocate_fe_matrix_and_vector(pop,r,x,ode_cache)

# ode_cache = update_cache!(ode_cache,op,r)
# RB.fe_matrix_and_vector!(A,b,pop,r,dtθ,x,ode_cache,y)

trialr = evaluate(trial,r)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,EvaluationFunction(trialr,y))
end
xh = TransientCellField(EvaluationFunction(trialr,y),dxh)
trial0 = evaluate(trial,nothing)
u = get_trial_fe_basis(trial0)
v = get_fe_basis(test)
assem = FEM.get_param_assembler(feop.op.assem,r)

i = 1
dc = feop.op.jacs[i](get_params(r),get_times(r),xh,u,v)
trian = first(feop.trian_jacs[i])
matdata = FEM.collect_cell_matrix_for_trian(trial0,test,dc,trian)
AA = allocate_matrix(assem,matdata)
