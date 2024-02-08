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

snaps,comp = RB.collect_solutions(rbinfo,fesolver,feop,uh0μ)
red_trial,red_test = reduced_fe_space(rbinfo,feop,snaps)

odeop = get_algebraic_operator(feop)
pop = GalerkinProjectionOperator(odeop,red_trial,red_test)
red_lhs,red_rhs = RB.reduced_matrix_vector_form(rbsolver,pop,snaps)
red_op = reduced_operator(pop,red_lhs,red_rhs)

s = RB.select_snapshots(snaps,1)
cache = RB.allocate_reduced_matrix_and_vector(rbsolver,red_op,s)
matvec_cache,coeff_cache,lincomb_cache = cache
A,b = RB.collect_matrices_vectors!(rbsolver,red_op,s,matvec_cache)
A_coeff,b_coeff = RB.mdeim_coeff!(coeff_cache,op.lhs,op.rhs,A,b)
A_red,b_red = RB.mdeim_lincomb!(lincomb_cache,op.lhs,op.rhs,A_coeff,b_coeff)


ids_all_time = RB._common_reduced_times(red_op)
sids = RB.select_snapshots(s,:,ids_all_time)
A,b = collect_matrices_vectors!(rbsolver,red_op.pop,sids,matvec_cache)
Aval = map(FEM.get_values,A)

aa = red_op.lhs[1][1]
ss = Aval[1][1]
ids_space = RB.get_indices_space(aa)
ids_time = RB.get_indices_time(aa)
corresponding_ids_time = filter(!isnothing,indexin(ids_all_time,ids_time))
# RB.tensor_getindex(ss,[1,7,4],corresponding_ids_time,1:num_params(s))
cols = RB.col_index(ss,corresponding_ids_time,1:num_params(ss))
getindex(s,ids_space,cols)
