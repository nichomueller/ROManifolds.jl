# imports
using Gridap
using jROM

# define a parametric function
a(x,μ,t) = sum(x)*sum(μ)*sin(t)
a(μ,t) = x -> a(x,μ,t)

# define transient parameter space
pdomain = [[0,1],[0,1]]
tdomain = 0:0.1:1
D = TransientParamSpace(pdomain,tdomain)

# extract a realization from D, for 10 different parameters
r = realization(D;nparams=10)

# define the values of a(⋅,μ,t) ∀ (μ,t) ∈ r
ar = TransientParamFunction(a,r)

# select a random x, where we evaluate ar
x = Point(1,1)
eval_ar = ar(x)
@assert isa(eval_ar,Vector{Float64})
@assert length(eval_ar) == num_params(r)*num_times(r) == 100

# verify the correctness of ar
eval_naive_a = []
for t in get_times(r), μ in get_params(r)
  push!(eval_naive_a,a(x,μ,t))
end

@assert eval_naive_a == eval_ar

# automatic differentiation
ar_t = ∂t(ar)
ar_x = ∇(ar)
eval_ar_t = ar_t(x)
eval_ar_x = ar_x(x)
@assert typeof(eval_ar_t) == typeof(eval_ar)
@assert length(eval_ar_t) == length(eval_ar)
@assert typeof(eval_ar_x) == Vector{VectorValue{2,Float64}}
@assert length(eval_ar_x) == length(eval_ar)

# define the derivatives by hand
a_t(x,μ,t) = sum(x)*sum(μ)*cos(t)
a_x(x,μ,t) = Point(x[2]*sum(μ)*sin(t),x[1]*sum(μ)*sin(t))

eval_naive_a_t = []
eval_naive_a_x = []
for t in get_times(r), μ in get_params(r)
  push!(eval_naive_a_t,a_t(x,μ,t))
  push!(eval_naive_a_x,a_x(x,μ,t))
end

@assert eval_naive_a_t == eval_ar_t
@assert eval_naive_a_x == eval_ar_x

# op = rbop.op_nonlinear.op.feop
# r = realization(op;nparams=1)
# x = zero_free_values(get_fe_trial(rbop)(r))
# ode_cache = allocate_cache(rbop,r)
# ode_cache = update_cache!(ode_cache,rbop,r)

# Xh, = ode_cache
# dxh = ()
# xh = TransientCellField(EvaluationFunction(Xh[1],x),dxh)
# # jacobians!(A,op.feop,r,xh,γ,ode_cache)
# trial = evaluate(get_trial(op.feop),nothing)
# test = get_test(op.feop)
# u = get_trial_fe_basis(trial)
# v = get_fe_basis(test)
# assem = FEM.get_param_assembler(op.feop.op.assem,r)
# trian = op.feop.trian_jacs[1][1]
# i = 1
# dcc = op.feop.op.jacs[i](get_params(r),get_times(r),xh,u,v)
# AAA = dcc[trian]
# matdata = FEM.collect_cell_matrix_for_trian(trial,test,dcc,trian)

#############################

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
g(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

induced_norm((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
_feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,induced_norm,ptspace,trial,test)
feop_nlin = FEOperatorWithTrian(_feop_nlin,trian_res,trian_jac)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
fesolver = ThetaMethod(nls,dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceOnlyMDEIM();nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)

using Serialization
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("navier_stokes","toy_mesh")))
fesnaps = Serialization.deserialize(RB.get_snapshots_filename(test_dir))
rbop = Serialization.deserialize(RB.get_op_filename(test_dir))

odeop_nlin = get_algebraic_operator(feop_nlin)
op_nlin = RB.PODOperator(odeop_nlin,get_trial(rbop),get_test(rbop))
s_mdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
r_mdeim = RB.get_realization(s_mdeim)
contribs_mat,contribs_vec = fe_jacobian_and_residual(rbsolver,op_nlin,s_mdeim)

using Gridap.CellData
red_lhs,red_rhs = RB.reduced_matrix_vector_form(rbsolver,op_nlin,fesnaps)
trians_rhs = get_domains(red_rhs)
trians_lhs = map(get_domains,red_lhs)
new_op_nlin = change_triangulation(op_nlin,trians_rhs,trians_lhs)
rbop_nlin = PODMDEIMOperator(new_op_nlin,red_lhs,red_rhs)

r = r_mdeim[1,:]
θ == 0.0 ? dtθ = dt : dtθ = dt*θ
red_test = get_test(rbop_nlin)
red_trial = get_trial(rbop_nlin)(r)
fe_trial = get_fe_trial(rbop_nlin)(r)
red_x = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
z = select_snapshots(fesnaps,:,:,1) |> get_values
@check typeof(y) == typeof(z)
s_mdeim_vals = get_values(s_mdeim)
@check z[1] ≈ s_mdeim_vals[1]
@check z[2] ≈ s_mdeim_vals[21]
@check z[3] ≈ s_mdeim_vals[41]

ode_cache = allocate_cache(rbop_nlin,r)
cache_nlin_jac,cache_nlin_res = ODETools._allocate_matrix_and_vector(rbop_nlin,r,y,ode_cache)
A_nlin = ODETools.jacobians!(cache_nlin_jac,rbop_nlin,r,(z,),(1,),ode_cache)
b_nlin = residual!(cache_nlin_res,rbop_nlin,r,(z,),ode_cache)

function get_fe_snaps(_r)
  r = copy(_r)
  FEM.shift_time!(r,dt*(θ-1))

  function mat(x,μ,t)
    g_t(x,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
    g_t(t) = x->g_t(x,t)
    fs_u = TrialFESpace(test_u,g_t(t))
    fs = MultiFieldFESpace([fs_u,trial_p];style=BlockMultiFieldStyle())

    xh = TransientCellField(EvaluationFunction(fs,x),())
    assemble_matrix((du,v)->dc(xh[1],du,v,dΩ),fs_u,test[1])
  end
  function vec(x,μ,t)
    g_t(x,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
    g_t(t) = x->g_t(x,t)
    fs_u = TrialFESpace(test_u,g_t(t))
    fs = MultiFieldFESpace([fs_u,trial_p];style=BlockMultiFieldStyle())

    xh = TransientCellField(EvaluationFunction(fs,x),())
    assemble_vector(v->c(xh[1],v,dΩ),test[1])
  end
  C = Snapshots(ParamArray([vec(z[i],μ,t) for (i,(μ,t)) in enumerate(r)]),r)
  dC = Snapshots(ParamArray([mat(z[i],μ,t) for (i,(μ,t)) in enumerate(r)]),r)
  C,dC
end

snapsC,snapsdC = get_fe_snaps(r)

mydC = select_snapshots(contribs_mat[1][Ω][1],:,:,1)
@check mydC ≈ snapsdC

myC = select_snapshots(contribs_vec[Ω][1],:,:,1)
@check myC ≈ snapsC

# compression error

dCrb = RB.compress(snapsdC,red_trial[1],red_test[1];combine=(x,y)->θ*x+(1-θ)*y)
Crb = RB.compress(snapsC,red_test[1])

# bad
norm(A_nlin[Block(1,1)][1] - dCrb) / norm(dCrb)
norm(b_nlin[Block(1)][1] - Crb) / norm(Crb)

# check at interpolation points

adM = red_lhs[1][1][1]
inds_space = adM.integration_domain.indices_space
inds_time = adM.integration_domain.indices_time

cache_nlin_jac,cache_nlin_res = ODETools._allocate_matrix_and_vector(rbop_nlin,r,y,ode_cache)
# A_nlin = ODETools.jacobians!(cache_nlin_jac,rbop_nlin,r,(z,),(1,),ode_cache)
fe_A,coeff_cache,lincomb_cache = cache_nlin_jac
fe_sA = fe_jacobians!(fe_A,rbop_nlin,r,(z,),(1,),ode_cache)
fe_sA1 = fe_sA[1][1][1]

snapsdC_idx = RB._select_snapshots_at_space_time_locations(snapsdC,adM,inds_time)
snapsdC_idx == fe_sA[1][1][1]


# heateq

adA = rbop.lhs[1][1]
inds_space = adA.integration_domain.indices_space
inds_time = adA.integration_domain.indices_time

r = realization(feop;nparams=1)
y = zero_free_values(get_fe_trial(rbop)(r))
z = zero_free_values(get_fe_trial(rbop)(r))
ode_cache = allocate_cache(rbop,r)
cache_mat,cache_vec = ODETools._allocate_matrix_and_vector(rbop,r,y,ode_cache)
fe_A,coeff_cache,lincomb_cache = cache_mat
ode_cache = update_cache!(ode_cache,rbop,r)
fe_sA = fe_jacobians!(fe_A,rbop,r,(y,z),(1,1/(dt*θ)),ode_cache)
fe_sA1 = fe_sA[1][1]

snapsA = Snapshots(ParamArray(
  [assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial(nothing),test) for (μ,t) in r
  ]),r)
snapsdA_idx = RB._select_snapshots_at_space_time_locations(snapsA,adA,inds_time)
snapsdA_idx ≈ fe_sA1

using Gridap.CellData
red_trian = get_domains(rbop.lhs[1])[1]

cell_dof_ids = get_cell_dof_ids(test,Ω)
indices_space_rows = fast_index(inds_space,num_free_dofs(test))
red_integr_cells = RB.get_reduced_cells(indices_space_rows,cell_dof_ids)
################################################################################
adH = rbop.rhs[1]
inds_spaceH = adH.integration_domain.indices_space
inds_timeH = adH.integration_domain.indices_time

# fe_b,coeff_cacheb,lincomb_cacheb = cache_vec
# fe_sb = fe_residual!(fe_b,rbop,r,(y,z),ode_cache)
# fe_sb1 = fe_sb[1]
cache_mat,cache_vec = ODETools._allocate_matrix_and_vector(rbop,r,y,ode_cache)
fe_b,coeff_cacheb,lincomb_cacheb = cache_vec
# red_r,red_times,red_xhF,red_ode_cache = RB._select_fe_quantities_at_time_locations(rbop.rhs,r,(y,z),ode_cache)
odeop = get_algebraic_operator(feop)
# b = residual!(fe_b,odeop,red_r,red_xhF,red_ode_cache)
b = residual!(fe_b,odeop,r,(y,z),ode_cache)
sb = Snapshots(b[1],r)
# bi = RB._select_snapshots_at_space_time_locations(sb,rbop.rhs,red_times)
bi = RB.reverse_snapshots_at_indices(sb,inds_spaceH)[inds_timeH]
fe_sb1 = bi[1]

ns = num_free_dofs(test)
function get_fe_snaps(_r)
  r = copy(_r)
  FEM.shift_time!(r,dt*(θ-1))

  function get_res(μ,t)
    g_t(x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
    g_t(t) = x->g_t(x,t)
    fs_t = TransientTrialFESpace(test,g_t)
    dfs_t = ∂t(fs_t)
    fs = fs_t(t)
    dfs = dfs_t(t)

    R(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ

    x0 = zeros(ns)
    xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
    assemble_vector(v->R(t,xh,v),test)
  end
  pR = ParamArray([get_res(μ,t) for (μ,t) in r])
  snapsR = Snapshots(pR,r)

  return snapsR
end
snapsH = get_fe_snaps(r)
snapsdH_idx = RB._select_snapshots_at_space_time_locations(snapsH,adH,inds_timeH)
snapsdH_idx ≈ fe_sb1


cell_dof_ids = get_cell_dof_ids(test,Ω)
indices_space_rows = fast_index(inds_spaceH,num_free_dofs(test))
red_integr_cells = RB.get_reduced_cells(indices_space_rows,cell_dof_ids)

for i in eachindex(cell_dof_ids)
  if !(i ∈ red_integr_cells)
    @check !(any([j ∈ cell_dof_ids[i] for j in indices_space_rows]))
  end
end
