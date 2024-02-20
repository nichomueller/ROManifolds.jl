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

snaps, = ode_solutions(rbsolver,feop,uh0μ)
# snaps = with_debug() do distribute
#   load_distributed_snapshots(distribute,info)
# end

function new_reduced_fe_space(
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

function new_compress(r::DistributedRBSpace,s::DistributedTransientSnapshots)
  map(local_views(r),local_views(s)) do r,s
    compress(r,s)
  end
end

new_trial,new_test = new_reduced_fe_space(info,feop,snaps)

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

basis_space = new_test.basis_space
basis_time = new_test.basis_time

# try to approximate in space only

b_rb_own = map(own_values(bsnap),own_values(basis_space)) do b,bs
  bs'*b[:,1]
end
A_rb_own_own = map(own_values(Asnap),own_values(Msnap),own_values(basis_space)) do A,M,bs
  bs'*(get_values(A)[1]+get_values(M)[1]/dt)*bs
end
matching_ghost_ids = map()
A_rb_own_ghost = map(
  own_ghost_values(Asnap),
  own_values(basis_space),
  ghost_values(basis_space),
  matching_ghost_ids) do A,M,bso,bsg,ids
  bso'*(get_values(A)[1]+get_values(M)[1][:,]/dt)*bsg[ids,:]
end

# try to approximate in space time

b_rb_own = map(own_values(bsnap),own_values(basis_space),basis_time) do b,bs,bt
  red_xmat = (bs'*b)*bt
  vec(red_xmat')
end

map(own_values(bsnap),own_values(basis_space),basis_time) do b,bs,bt
  red_xmat = (bs'*b)*bt
  vec(red_xmat')
end

r0 = FEM.TransientParamRealizationAt(ParamRealization([[1,2,3]]),Base.RefValue(dt))
w0 = get_free_dof_values(uh0μ(get_params(r0)))
wf = copy(w0)
sol = GenericODEParamSolution(fesolver,get_algebraic_operator(feop.op),w0,r0)
# solve_step!(wf,fesolver,sol.op,r0,w0,nothing)

ode_cache = allocate_cache(sol.op,r0)
vθ = similar(w0)
vθ .= 0.0
l_cache = nothing
A,b = ODETools._allocate_matrix_and_vector(sol.op,r0,w0,ode_cache)
ode_cache = update_cache!(ode_cache,sol.op,r0)
ODETools._matrix_and_vector!(A,b,sol.op,r0,dtθ,w0,ode_cache,vθ)
afop = AffineOperator(A,b)
newmatrix = true
l_cache = solve!(wf,fesolver.nls,afop,l_cache,newmatrix)
