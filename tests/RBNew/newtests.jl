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
# model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
# model = DiscreteModelFromFile(model_dir)

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
# Γn = BoundaryTriangulation(model,tags=["neumann"])
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
# test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
info = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,st_mdeim=true)

rbsolver = RBSolver(info,fesolver)

snaps,comp = RB.fe_solutions(rbsolver,feop,uh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)

son = select_snapshots(snaps,RB.online_params(info))
ron = get_realization(son)
xrb, = solve(rbsolver,red_op,ron)
son_rev = reverse_snapshots(son)
norm(xrb - son_rev)

info_space = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
rbsolver_space = RBSolver(info_space,fesolver)
red_op_space = reduced_operator(rbsolver_space,feop,snaps)
xrb_space, = solve(rbsolver_space,red_op_space,ron)
norm(xrb_space - son_rev)

results = solve(rbsolver,feop,uh0μ)

# CORRECT ST-MDEIM
info = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,st_mdeim=true)
rbsolver = RBSolver(info,fesolver)

θ == 0.0 ? dtθ = dt : dtθ = dt*θ

r = ron
op = red_op
red_trial = get_trial(op)(r)
fe_trial = get_fe_trial(op)(r)
red_x = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

ode_cache = allocate_cache(op,r)
mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(op,r,y,ode_cache)
ode_cache = update_cache!(ode_cache,op,r)
fe_A,coeff_cache,lincomb_cache = vec_cache
fe_sA = fe_vector!(fe_A,op,r,(y,z),ode_cache)

red_times = RB._union_reduced_times_vec(op)
red_r = r[:,red_times]
red_xhF,red_ode_cache = RB._select_fe_quantities_at_time_locations((y,z),ode_cache,r,red_times)
A = fe_vector!(fe_A,op.pop,red_r,red_xhF,red_ode_cache)
# Ai = RB._select_snapshots_at_space_time_locations(A,op.rhs,red_times)

# B = get_values(op.rhs)[2]
# ids_space = RB.get_indices_space(B)
# ids_time = filter(!isnothing,indexin(red_times, RB.get_indices_time(B)))
# ids_param = Base.OneTo(num_params(get_values(A)[1]))
# snew = RB.reverse_snapshots_at_indices(get_values(A)[1],ids_space)
# ye = select_snapshots(snew,ids_time,ids_param)

# fe_sA1 = get_values(fe_sA[1])[1]
# coeff,coeff_recast = get_values(coeff_cache[1][1])[1],get_values(coeff_cache[1][2])[1]
# lhs_ad1 = get_values(op.lhs[1])[1]
# mdeim_interpolation = lhs_ad1.mdeim_interpolation
# ns = num_reduced_space_dofs(lhs_ad1)
# nt = num_reduced_times(lhs_ad1)
# np = length(coeff_recast)
A_coeff = mdeim_coeff!(coeff_cache,op.rhs,fe_sA)
mdeim_lincomb!(lincomb_cache,op.rhs,A_coeff)

YE = get_values(A_coeff)[1]

# bvec = reshape(fe_sA1,:,np)
# ldiv!(coeff,mdeim_interpolation,bvec)
# for j in 1:ns
#   sorted_idx = [(i-1)*ns+j for i = 1:nt]
#   @inbounds for i = eachindex(coeff_recast)
#     coeff_recast[i][:,j] = lhs_ad1.basis_time*coeff[sorted_idx,i]
#   end
# end


# SPACE ONLY
info_space = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
rbsolver_space = RBSolver(info_space,fesolver)
op_space = reduced_operator(rbsolver_space,feop,snaps)

ode_cache_space = allocate_cache(op_space,r)
mat_cache_space,vec_cache_space = ODETools._allocate_matrix_and_vector(op_space,r,y,ode_cache)
ode_cache_space = update_cache!(ode_cache_space,op_space,r)
fe_A_space,coeff_cache_space,lincomb_cache_space = vec_cache_space
fe_sA_space = fe_vector!(fe_A_space,op_space,r,(y,z),ode_cache_space)

# red_times_space = RB._union_reduced_times_vec(op_space)
# red_r_space = r[:,red_times_space]
# red_xhF_space,red_ode_cache_space = RB._select_fe_quantities_at_time_locations(
#   (y,z),ode_cache_space,r,red_times_space)
# A_space = fe_vector!(fe_A_space,op_space.pop,red_r_space,red_xhF_space,red_ode_cache_space)
# Ai_space = RB._select_snapshots_at_space_time_locations(A_space,op_space.rhs,red_times_space)

A_coeff_space = mdeim_coeff!(coeff_cache_space,op_space.rhs,fe_sA_space)
mdeim_lincomb!(lincomb_cache_space,op_space.rhs,A_coeff_space)

YE_OK = get_values(A_coeff_space)[1]

YE_DIFF = YE - YE_OK
map(YE_DIFF) do x
  norm(x)
end
