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

red_mat = RB.reduced_matrix_form(rbsolver,pop,contribs_mat)
red_vec = RB.reduced_vector_form(rbsolver,pop,contribs_vec)

s = contribs_vec[Ω]

basis_space = map(local_views(pop.test)) do test
  test.basis_space
end

########
function PartitionedArrays.p_sparse_matrix_cache_impl(
  ::Type,matrix_partition,row_partition,col_partition)

  function setup_snd(part,parts_snd,row_indices,col_indices,values)
      local_row_to_owner = local_to_owner(row_indices)
      local_to_global_row = local_to_global(row_indices)
      local_to_global_col = local_to_global(col_indices)
      owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
      ptrs = zeros(Int32,length(parts_snd)+1)
      for (li,lj,v) in PartitionedArrays.nziterator(values)
          owner = local_row_to_owner[li]
          if owner != part
              ptrs[owner_to_i[owner]+1] +=1
          end
      end
      Algebra.length_to_ptrs!(ptrs)
      println(ptrs)
      k_snd_data = zeros(Int32,ptrs[end]-1)
      gi_snd_data = zeros(Int,ptrs[end]-1)
      gj_snd_data = zeros(Int,ptrs[end]-1)
      for (k,(li,lj,v)) in enumerate(nziterator(values))
          owner = local_row_to_owner[li]
          if owner != part
              p = ptrs[owner_to_i[owner]]
              k_snd_data[p] = k
              gi_snd_data[p] = local_to_global_row[li]
              gj_snd_data[p] = local_to_global_col[lj]
              ptrs[owner_to_i[owner]] += 1
          end
      end
      Algebra.rewind_ptrs!(ptrs)
      k_snd = JaggedArray(k_snd_data,ptrs)
      gi_snd = JaggedArray(gi_snd_data,ptrs)
      gj_snd = JaggedArray(gj_snd_data,ptrs)
      k_snd, gi_snd, gj_snd
  end
  function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
      global_to_local_row = global_to_local(row_indices)
      global_to_local_col = global_to_local(col_indices)
      ptrs = gi_rcv.ptrs
      k_rcv_data = zeros(Int32,ptrs[end]-1)
      for p in 1:length(gi_rcv.data)
          gi = gi_rcv.data[p]
          gj = gj_rcv.data[p]
          li = global_to_local_row[gi]
          lj = global_to_local_col[gj]
          k = PartitionedArrays.nzindex(values,li,lj)
          @boundscheck @assert k > 0 "The sparsity pattern of the ghost layer is inconsistent"
          k_rcv_data[p] = k
      end
      k_rcv = JaggedArray(k_rcv_data,ptrs)
      k_rcv
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  k_snd, gi_snd, gj_snd = map(setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  gi_rcv = PartitionedArrays.exchange_fetch(gi_snd,graph)
  gj_rcv = PartitionedArrays.exchange_fetch(gj_snd,graph)
  k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition)
  buffers = map(PartitionedArrays.assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
  cache = map(PartitionedArrays.VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
  map(PartitionedArrays.SparseMatrixAssemblyCache,cache)
end
