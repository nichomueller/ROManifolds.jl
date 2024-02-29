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
using Gridap.Algebra
using Gridap.CellData
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

# new_trial,new_test = Distributed.new_reduced_fe_space(info,feop,snaps)

sk = select_snapshots(snaps,first(RB.online_params(info)))
pk = get_values(sk)

pk_rb = compress(red_test,sk)
pk_rec = recast(red_trial(get_realization(sk)),pk_rb)

norm(pk_rec - pk) / norm(pk)

odeop = get_algebraic_operator(feop)
op = RBOperator(odeop,red_trial,red_test)
# red_lhs,red_rhs = reduced_matrix_vector_form(rbsolver,op,snaps)

θ == 0.0 ? dtθ = dt : dtθ = dt*θ
smdeim = select_snapshots(snaps,RB.mdeim_params(info))
x = get_values(smdeim)
r = get_realization(smdeim)

y = similar(x)
y .= 0.0
ode_cache = allocate_cache(op,r)
A,b = allocate_fe_jacobian_and_residual(op,r,x,ode_cache)

ode_cache = update_cache!(ode_cache,op,r)
contribs_mat,contribs_vec = fe_jacobian_and_residual!(A,b,op,r,dtθ,x,ode_cache,y)

# red_mat = RB.reduced_matrix_form(rbsolver,op,contribs_mat)
# red_vec = RB.reduced_vector_form(rbsolver,op,contribs_vec)

c = distributed_array_contribution()
trian,vals = get_domains(contribs_vec)[1],get_values(contribs_vec)[1]
# RB.reduced_vector_form!(c,info,op,values,trian)
basis_space,basis_time = reduced_basis(vals;ϵ=RB.get_tol(info))
# lu_interp,red_trian,integration_domain = mdeim(info,fs,trian,basis_space,basis_time)
lu_interp,red_trian,integration_domain = map(
    local_views(test),
    local_views(trian),
    local_views(basis_space),
    local_views(basis_time)) do fs,trian,basis_space,basis_time

  mdeim(info,fs,trian,basis_space,basis_time)
end |> tuple_of_arrays
# proj_basis_space = compress_basis_space(basis_space,red_test)
basis_test = get_basis_space(red_test)
basis_test'*basis_space
comb_basis_time = combine_basis_time(red_test)


# dummy test for online phase, no mdeim
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

b_rb = compress(red_test,bsnap)
A_rb = compress(red_trial(ron),red_test,Asnap;combine=(x,y)->θ*x+(1-θ)*y) # see below
M_rb = compress(red_trial(ron),red_test,Msnap;combine=(x,y)->θ*(x-y))  # see below
AM_rb = map(+,A_rb,M_rb)

x_rb = map(\,AM_rb,b_rb)

x_rec = recast(red_trial(ron),x_rb)
norm(x_rec + x) / norm(x)

# b_rec = recast(red_trial(ron),b_rb)
# norm(b_rec + b) / norm(b)

x_rec = map(own_values(b),own_values(A),own_values(M),local_views(red_test)) do b,A,M,test
  basis_space,basis_time = test.basis_space,test.basis_time
  bsnap = Snapshots(b,ron)
  Asnap = Snapshots(A,ron)
  Msnap = Snapshots(M,ron)
  b_rb = compress(test,bsnap)
  A_rb = compress(test,test,Asnap;combine=(x,y)->θ*x+(1-θ)*y) # see below
  M_rb = compress(test,test,Msnap;combine=(x,y)->θ*(x-y))
  AM_rb = A_rb+M_rb
  x_rb = -AM_rb \ b_rb
  recast(test,x_rb)
end

function RB.compress_basis_space(A::AbstractMatrix,trial::RBSpace,test::RBSpace)
  basis_test = get_basis_space(test)
  basis_trial = get_basis_space(trial)
  vals = vec(get_values(A))
  map(vals) do A
    basis_test'*A*basis_trial
  end
end

function _norm_space_time(x::PVector,xrb::PVector)
  err_norm_contribs,sol_norm_contribs = map(own_values(x),own_values(xrb)) do x,xrb
    norm(x-xrb)^2,norm(x)^2
  end |> tuple_of_arrays
  err_norm = reduce(+,err_norm_contribs;init=zero(eltype(err_norm_contribs)))^(1/2)
  sol_norm = reduce(+,sol_norm_contribs;init=zero(eltype(sol_norm_contribs)))^(1/2)
  norm(err_norm)/norm(sol_norm)
end

# try with ghost values
function ghost_reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  # dtrial = _to_distributed_fe_space(trial)
  test = get_test(feop)
  soff = select_snapshots(s,RB.offline_params(info))
  basis_space,basis_time = map(ghost_values(soff)) do s
    reduced_basis(s,nothing;ϵ=RB.get_tol(info))
  end |> tuple_of_arrays

  reduced_trial = RBSpace(trial,basis_space,basis_time)
  reduced_test = RBSpace(test,basis_space,basis_time)
  return reduced_trial,reduced_test
end
# function ghost_compress(r::DistributedRBSpace,s::DistributedTransientSnapshots)
#   map(local_views(r),ghost_values(s)) do r,s
#     compress(r,s)
#   end
# end
# function ghost_compress(
#   trial::DistributedRBSpace,
#   test::DistributedRBSpace,
#   s::DistributedTransientSnapshots;
#   kwargs...)

#   map(local_views(trial),local_views(test),ghost_values(s)) do trial,test,s
#     compress(trial,test,s;kwargs...)
#   end
# end

ghost_red_trial,ghost_red_test = ghost_reduced_fe_space(info,feop,snaps)

function own_ghost_compress(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  s::DistributedTransientSnapshots;
  kwargs...)

  map(local_views(trial),local_views(test),own_ghost_values(s)) do trial,test,s
    compress(trial,test,s;kwargs...)
  end
end

ghost_A_rb = own_ghost_compress(red_trial(ron),ghost_red_test,Asnap;combine=(x,y)->θ*x+(1-θ)*y)
ghost_M_rb = own_ghost_compress(red_trial(ron),ghost_red_test,Msnap;combine=(x,y)->θ*(x-y))

AA = Asnap.snaps.matrix_partition.items[1]
basis_space_test = ghost_red_test.basis_space.items[1]
basis_space_trial = red_trial.basis_space.items[1]

#############################

function _reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  soff = select_snapshots(s,RB.offline_params(info))
  basis_space,basis_time = map(local_values(soff)) do s
    reduced_basis(s,nothing;ϵ=RB.get_tol(info))
  end |> tuple_of_arrays

  reduced_trial = RBSpace(trial,basis_space,basis_time)
  reduced_test = RBSpace(test,basis_space,basis_time)
  return reduced_trial,reduced_test
end

function _change_ghost(a::PMatrix{T},ids::PRange;is_consistent=false,make_consistent=true) where T
  same_partition = (a.row_partition === partition(ids))
  a_new = same_partition ? a : _change_ghost(T,a,ids)
  if make_consistent && (!same_partition || !is_consistent)
    _consistent!(a_new) |> wait
  end
  return a_new
end

function _change_ghost(::Type{<:AbstractMatrix},a::PMatrix,ids::PRange)
  a_new = similar(a,eltype(a),(ids,PRange(a.col_partition)))
  # Equivalent to copy!(a_new,a) but does not check that owned indices match
  map(copy!,own_values(a_new),own_values(a))
  return a_new
end

function _consistent!(a::PMatrix)
  insert(a,b) = b
  cache = map(reverse,a.cache)
  t = PartitionedArrays.assemble!(insert,partition(a),cache)
  @async begin
    wait(t)
    a
  end
end

new_basis_space,new_basis_time = new_test.basis_space,new_test.basis_time
Bsnap = _change_ghost(bsnap.snaps,test.gids)

map(local_views(Bsnap),new_basis_space) do b,bs
  norm(b - bs*bs'*b) / norm(b)
end
