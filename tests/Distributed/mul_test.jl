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
rbsolver = RBSolver(fesolver,dir;nparams_state=10,nsnaps_mdeim=5,nparams_test=5,save_structures=false)
info = rbsolver.info

snaps = with_debug() do distribute
  load_distributed_snapshots(distribute,info)
end

function _reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  soff = select_snapshots(s,RB.offline_params(info))
  row_partition = s.snaps.row_partition
  basis_space,basis_time = map(local_values(soff)) do s
    reduced_basis(s,nothing;ϵ=RB.get_tol(info))
  end |> tuple_of_arrays
  col_partition = Distributed.get_col_partition(basis_space,row_partition)
  p_basis_space = PMatrix(basis_space,row_partition,col_partition)
  reduced_trial = RBSpace(trial,p_basis_space,basis_time)
  reduced_test = RBSpace(test,p_basis_space,basis_time)
  return reduced_trial,reduced_test
end

red_trial,red_test = _reduced_fe_space(info,feop,snaps)

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

function _compress_space(basis_space1,basis_space2,A)
  vals = vec(get_values(A))
  red_xvec = map(vals) do A
    basis_space1'*A*basis_space2
  end
  stack(vec.(red_xvec))'
end

function _compress_space(basis_space1::PMatrix,basis_space2::PMatrix,A::DistributedTransientSnapshots)
  own_own = map(own_values(basis_space1),own_values(basis_space2),own_values(A)) do bs1,bs2,A
    _compress_space(bs1,bs2,A)
  end
  bs2_compatible=GridapDistributed.change_ghost(basis_space2,axes(A,2);make_consistent=true)
  own_ghost = map(own_values(basis_space1),ghost_values(bs2_compatible),own_ghost_values(A)) do bs1,bs2,A
    _compress_space(bs1,bs2,A)
  end
  map(+,own_own,own_ghost)
end

function _compress_time(basis_time1,basis_time2,A;combine=(x,y)->x)
  T = Float64
  ns1,ns2 = 2,2
  nt1 = size(basis_time1,2)
  nt2 = size(basis_time2,2)
  st_proj = zeros(T,nt1,nt2,ns1*ns2)
  st_proj_shift = zeros(T,nt1,nt2,ns1*ns2)
  @inbounds for ins = 1:ns1*ns2, jt = 1:nt2, it = 1:nt1
    st_proj[it,jt,ins] = sum(basis_time1[:,it].*basis_time2[:,jt].*A[:,ins])
    st_proj_shift[it,jt,ins] = sum(basis_time1[2:end,it].*basis_time2[1:end-1,jt].*A[2:end,ins])
  end
  st_proj = combine(st_proj,st_proj_shift)
  st_proj_mat = zeros(T,ns1*nt1,ns2*nt2)
  @inbounds for i = 1:ns2, j = 1:ns1
    st_proj_mat[1+(j-1)*nt1:j*nt1,1+(i-1)*nt2:i*nt2] = st_proj[:,:,(i-1)*ns1+j]
  end
  return st_proj_mat
end

function _compress_space_time(trial,test,A::DistributedTransientSnapshots;kwargs...)
  A_space = _compress_space(test.basis_space,trial.basis_space,A)
  map(test.basis_time,trial.basis_time,A_space) do bt1,bt2,A
    _compress_time(bt1,bt2,A;kwargs...)
  end
end

function _compress_space_time(test,b::DistributedTransientSnapshots)
  map(own_values(test.basis_space),test.basis_time,own_values(b)) do bs,bt,b
    m = (bs'*b)*bt
    vec(m')
  end
end

b_rb = _compress_space_time(red_test,bsnap)
A_rb = _compress_space_time(red_trial,red_test,Asnap;combine=(x,y)->θ*x+(1-θ)*y)
M_rb = _compress_space_time(red_trial,red_test,Msnap;combine=(x,y)->θ*(x-y))

x_rb_own = map(A_rb,M_rb,b_rb,own_values(red_test.basis_space),red_test.basis_time
  ) do a,m,b,bs,bt

  x_rb = -(a+m) \ b
  x_mat_rb = reshape(x_rb,size(bt,2),size(bs,2))
  x_rec = bs*(bt*x_mat_rb)'
end

# x_rec = similar(son.snaps)
# map(copy!,own_values(x_rec),x_rb_own)
# consistent!(x_rec) |> fetch

err = map(A_rb,M_rb,b_rb,own_values(son),own_values(red_test.basis_space),red_test.basis_time
  ) do a,m,b,x_mat,bs,bt

  x_rb = -(a+m) \ b
  x_mat_rb = reshape(x_rb,size(bt,2),size(bs,2))
  x_rec = bs*(bt*x_mat_rb)'
  norm(x_rec - x_mat) / norm(x_mat)
end
