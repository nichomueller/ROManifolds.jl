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

ode_cache = update_cache!(ode_cache,pop,r)
contribs_mat,contribs_vec = fe_matrix_and_vector!(A,b,pop,r,dtθ,x,ode_cache,y)

# red_mat = RB.reduced_matrix_form(rbsolver,pop,contribs_mat)
# red_vec = RB.reduced_vector_form(rbsolver,pop,contribs_vec)

c = distributed_array_contribution()
trian,vals = get_domains(contribs_vec)[1],get_values(contribs_vec)[1]
# RB.reduced_vector_form!(c,info,pop,values,trian)
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
_b = PMatrix(bsnap.snaps.vector_partition,b.index_partition)

b_rb = compress(red_test,_b)
A_rb = compress(red_trial,red_test,A;combine=(x,y)->θ*x+(1-θ)*y)
M_rb = compress(red_trial,red_test,M;combine=(x,y)->θ*(x-y))
AM_rb = A_rb+M_rb

x_rb = AM_rb \ b_rb

x_rec = recast(red_trial,x_rb)
norm(x_rec - x) / norm(x)

# b_rb = compress(red_test,_b)
basis_space = get_basis_space(red_test)
# basis_space'*_b
A,B = basis_space',_b
Ta = eltype(A)
Tb = eltype(B)
T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
c = PMatrix{Matrix{T}}(undef,partition(axes(A,1)),partition(axes(B,2)))
fill!(c,zero(T))
# a_in_main = PartitionedArrays.to_trivial_partition(A)
row_partition_in_main=PartitionedArrays.trivial_partition(partition(axes(A,1)))
col_partition_in_main=PartitionedArrays.trivial_partition(partition(axes(A,2)))
destination = 1
T = eltype(A)
a_in_main = similar(A,T,PRange(row_partition_in_main),PRange(col_partition_in_main))
fill!(a_in_main,zero(T))
map(own_values(A.parent),partition(a_in_main),partition(axes(A,1)),partition(axes(A,2))) do aown,my_a_in_main,row_indices,col_indices
  if part_id(row_indices) == part_id(col_indices) == destination
    my_a_in_main[own_to_global(row_indices),own_to_global(col_indices)] .= aown'
  else
    my_a_in_main .= aown'
  end
end
# assemble!(a_in_main)
