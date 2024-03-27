#############
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

np = 3
nt = 10

pranges = fill([0,1],3)
tdomain = 0:nt
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1)
trial = TrialFESpace(test,x->0)

ns = num_free_dofs(test)

Φ = Float64.(I(ns))
Ψ = Float64.(I(10))
red_test = RBSpace(test,Φ,Ψ)
red_trial = RBSpace(trial,Φ,Ψ)

biform(u,v) = ∫(∇(v)⋅∇(u))dΩ
liform(v) = ∫(v)dΩ

_A = assemble_matrix(biform,trial,test)
_b = assemble_vector(liform,test)
snaps_biform = Snapshots(ParamArray([copy(_A) for _ = 1:nt*np]),r)
snaps_liform = Snapshots(ParamArray([copy(_b) for _ = 1:nt*np]),r)

@test norm(snaps_biform - hcat([_A.nzval for _ = 1:nt*np]...)) ≈ 0.0

np_online = 1
r_online = realization(ptspace,nparams=np_online)
snaps_biform_online = Snapshots(ParamArray([copy(_A) for _ = 1:nt]),r_online)
snaps_liform_online = Snapshots(ParamArray([copy(_b) for _ = 1:nt]),r_online)

# matrix

mdeim_style = RB.SpaceTimeMDEIM()#RB.SpaceOnlyMDEIM()
basis_space,basis_time = RB.reduced_basis(snaps_biform)
@check size(basis_space,2) == 1 && size(basis_time,2) == 1

indices_space = RB.get_mdeim_indices(basis_space)
interp_basis_space = view(basis_space,indices_space,:)
@check length(indices_space) == 1 && indices_space[1] == argmax(abs.(_A.nzval))

indices_time,lu_interp = RB._time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
if mdeim_style == RB.SpaceOnlyMDEIM()
  @check indices_time == get_times(r)
else
  @check length(indices_time) == 1
end
recast_indices_space = RB.recast_indices(basis_space,indices_space)
integration_domain = RB.ReducedIntegrationDomain(recast_indices_space,indices_time)
proj_basis_space = map(basis_space.values) do a
  Φ'*a*Φ
end
@check basis_space.values[1] ≈ proj_basis_space[1]

comb_basis_time = zeros(Float64,size(Ψ,1),size(Ψ,2),size(Ψ,2))
@inbounds for jt = axes(Ψ,2), it = axes(Ψ,2)
  comb_basis_time[:,it,jt] .= Ψ[:,it].*Ψ[:,jt]
end

A = AffineDecomposition(
  mdeim_style,proj_basis_space,basis_time,lu_interp,integration_domain,comb_basis_time)

coeff_cache = RB.allocate_mdeim_coeff(A,r_online)
lincomb_cache = RB.allocate_mdeim_lincomb(red_trial,red_test,r_online)

snaps_mat = RB._select_snapshots_at_space_time_locations(snaps_biform_online,A,indices_time)
RB.mdeim_coeff!(coeff_cache,A,snaps_mat)
_,coeff_mat = coeff_cache
RB.mdeim_lincomb!(lincomb_cache,A,coeff_mat)
_,_,mat_red = lincomb_cache

red_mat_snaps = RB.compress(snaps_biform_online,red_trial,red_test)
err = red_mat_snaps - mat_red[1]

@check norm(err)/sqrt(length(err)) ≤ 1e-12

# vector

mdeim_style = RB.SpaceTimeMDEIM() # RB.SpaceOnlyMDEIM() #
basis_space,basis_time = RB.reduced_basis(snaps_liform)
@check size(basis_space,2) == 1 && size(basis_time,2) == 1

indices_space = RB.get_mdeim_indices(basis_space)
interp_basis_space = view(basis_space,indices_space,:)
@check length(indices_space) == 1 && indices_space[1] == argmax(abs.(_b))

indices_time,lu_interp = RB._time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
if mdeim_style == RB.SpaceOnlyMDEIM()
  @check indices_time == get_times(r)
else
  @check length(indices_time) == 1
end

integration_domain = RB.ReducedIntegrationDomain(indices_space,indices_time)
proj_basis_space = map(eachcol(basis_space)) do a
  Φ'*a
end
@check basis_space ≈ proj_basis_space[1]

comb_basis_time = Ψ

b = AffineDecomposition(
  mdeim_style,proj_basis_space,basis_time,lu_interp,integration_domain,comb_basis_time)

coeff_cache = RB.allocate_mdeim_coeff(b,r_online)
lincomb_cache = RB.allocate_mdeim_lincomb(red_test,r_online)

snaps_vec = RB._select_snapshots_at_space_time_locations(snaps_liform_online,b,indices_time)
@check all(abs.(snaps_vec) .≈ maximum(abs.(snaps_liform_online)))

RB.mdeim_coeff!(coeff_cache,b,snaps_vec)
_,coeff_vec = coeff_cache
RB.mdeim_lincomb!(lincomb_cache,b,coeff_vec)
_,_,vec_red = lincomb_cache

red_vec_snaps = RB.compress(snaps_liform_online,red_test)
err = red_vec_snaps - vec_red[1]

@check norm(err)/sqrt(length(err)) ≤ 1e-12

#############

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# domain = (0,1,0,1)
# partition = (2,2)
# model = CartesianDiscreteModel(domain,partition)
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)
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

u0(x,μ) = 0.0
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
trial0 = trial(nothing)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
info = RBInfo(dir;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)

rbsolver = RBSolver(info,fesolver)

snaps,comp = RB.fe_solutions(rbsolver,feop,uh0μ)
red_trial,red_test = reduced_fe_space(info,feop,snaps)
odeop = get_algebraic_operator(feop)
op = RBOperator(odeop,red_trial,red_test)

ns = num_free_dofs(test)

# Φ = zeros(ns,2)
# Φ[1,1] = 1
# Φ[2,2] = 1
# Ψ = zeros(nt,2)
# Ψ[1,1] = 1
# Ψ[2,2] = 1
# red_test = RBSpace(test,Φ,Ψ)
# red_trial = RBSpace(trial,Φ,Ψ)

function get_fe_snaps(_r)
  # full order matrix
  r = copy(_r)
  shift!(r,dt*(θ-1))
  trial0 = trial(nothing)
  pA = ParamArray([assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial0,test) for (μ,t) in r])
  pM = ParamArray([assemble_matrix((u,v)->∫(v*u)dΩ,trial0,test)/(θ*dt) for (μ,t) in r])
  snapsA = Snapshots(pA,r)
  snapsM = Snapshots(pM,r)

  # full order vector

  function get_res(μ,t)
    g_t(x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
    g_t(t) = x->g_t(x,t)
    fs_t = TransientTrialFESpace(test,g_t)
    dfs_t = ∂t(fs_t)
    fs = fs_t(t)
    dfs = dfs_t(t)

    R(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn

    x0 = zeros(ns)
    xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
    assemble_vector(v->R(t,xh,v),test)
  end
  pR = ParamArray([get_res(μ,t) for (μ,t) in r])
  snapsR = Snapshots(pR,r)

  return snapsA,snapsM,snapsR
end

s_mdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
r_mdeim = RB.get_realization(s_mdeim)
contribs_mat,contribs_vec = jacobian_and_residual(rbsolver,op,s_mdeim)
snapsA,snapsM,snapsR = get_fe_snaps(r_mdeim)
@check contribs_mat[1][Ω] ≈ snapsA
@check contribs_mat[2][Ω] ≈ snapsM
@check contribs_vec[Ω] + contribs_vec[Γn] ≈ -snapsR

red_lhs,red_rhs = RB.reduced_matrix_vector_form(rbsolver,op,snaps)
red_op = reduced_operator(op,red_lhs,red_rhs)

snaps_on = RB.select_snapshots(snaps,1)
r_on = RB.get_realization(snaps_on)
snapsA,snapsM,snapsR = get_fe_snaps(r_on)

Arb = RB.compress(snapsA,red_trial,red_test;combine=(x,y)->θ*x+(1-θ)*y)
Mrb = RB.compress(snapsM,red_trial,red_test;combine=(x,y)->θ*(x-y))
AMrb = Arb + Mrb

Rrb = RB.compress(snapsR,red_test)

xrb = compress(snaps_on,get_test(red_op))
man_xrb = AMrb \ Rrb

# rb matrix/vector
θ == 0.0 ? dtθ = dt : dtθ = dt*θ
red_test = get_test(red_op)
red_trial = get_trial(red_op)(r_on)
fe_trial = trial(r_on)
red_x = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

snapsA,snapsM,snapsR = get_fe_snaps(r_on)
is_A = get_values(red_op.lhs[1])[1].integration_domain.indices_space
it_A = get_values(red_op.lhs[1])[1].integration_domain.indices_time
is_M = get_values(red_op.lhs[2])[1].integration_domain.indices_space
it_M = get_values(red_op.lhs[2])[1].integration_domain.indices_time
# is_R1 = get_values(red_op.rhs)[1].integration_domain.indices_space
# it_R1 = get_values(red_op.rhs)[1].integration_domain.indices_time
# is_R2 = get_values(red_op.rhs)[2].integration_domain.indices_space
# it_R2 = get_values(red_op.rhs)[2].integration_domain.indices_time

fe_A, = allocate_jacobian(red_op,r_on,y,ode_cache)
fe_sA = fe_jacobian!(fe_A,red_op,r_on,(y,z),(1,1/dtθ),ode_cache)
@check get_values(fe_sA[1])[1] ≈ stack(map(i->RB.get_values(snapsA)[i][is_A],it_A))
@check get_values(fe_sA[2])[1] ≈ stack(map(i->RB.get_values(snapsM)[i][is_M],it_M))

ode_cache = allocate_cache(red_op,r_on)
nl_cache = nothing
mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(red_op,r_on,y,ode_cache)
ode_cache = update_cache!(ode_cache,red_op,r_on)
A,b = ODETools._matrix_and_vector!(mat_cache,vec_cache,red_op,r_on,dtθ,y,ode_cache,z)

# errors
errA = norm(A[1] - AMrb)
errb = norm(b[1] + Rrb)
