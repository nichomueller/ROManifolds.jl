# takes into account only owned values

const DistributedRBSpace = RBSpace{S,BS,BT} where {S<:DistributedFESpace,BS,BT}

function RB.num_space_dofs(r::DistributedRBSpace)
  Ns = map(RB.num_space_dofs,local_views(r))
  PartitionedArrays.getany(Ns)
end

function RB.num_reduced_space_dofs(r::DistributedRBSpace)
  ns = map(RB.num_reduced_space_dofs,local_views(r))
  PartitionedArrays.getany(ns)
end

function FEM.num_times(r::DistributedRBSpace)
  Nt = map(num_times,local_views(r))
  PartitionedArrays.getany(Nt)
end

function RB.num_reduced_times(r::DistributedRBSpace)
  nt = map(RB.num_reduced_times,local_views(r))
  PartitionedArrays.getany(nt)
end

FESpaces.get_free_dof_ids(r::DistributedRBSpace) = get_free_dof_ids(r.space)

function RB.reduced_fe_space(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  # dtrial = _to_distributed_fe_space(trial)
  test = get_test(feop)
  soff = select_snapshots(s,RB.offline_params(solver))
  basis_space,basis_time = map(own_values(soff)) do s
    reduced_basis(s,nothing;ϵ=RB.get_tol(solver))
  end |> tuple_of_arrays

  reduced_trial = RBSpace(trial,basis_space,basis_time)
  reduced_test = RBSpace(test,basis_space,basis_time)
  return reduced_trial,reduced_test
end

function new_reduced_fe_space(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  T = eltype(s.snaps)
  trial = get_trial(feop)
  test = get_test(feop)
  soff = select_snapshots(s,RB.offline_params(solver))
  row_partition = s.snaps.row_partition
  basis_space,basis_time = map(own_values(soff)) do s
    reduced_basis(s,nothing;ϵ=RB.get_tol(solver))
  end |> tuple_of_arrays
  col_partition = get_col_partition(basis_space,row_partition)
  p_basis_space = PMatrix{Matrix{T}}(undef,row_partition,col_partition)
  map(copy!,own_values(p_basis_space),basis_space)
  consistent!(p_basis_space) |> wait
  reduced_trial = RBSpace(trial,p_basis_space,basis_time)
  reduced_test = RBSpace(test,p_basis_space,basis_time)
  return reduced_trial,reduced_test
end

function RB.reduced_basis(s::DistributedTransientSnapshots,args...;kwargs...)
  map(own_values(s)) do s
    reduced_basis(s,args...;kwargs...)
  end |> tuple_of_arrays
end

function FESpaces.get_vector_type(r::DistributedRBSpace)
  change_length(x) = x
  function change_length(::Type{<:PVector{<:ParamVector{T,A,L},B,C,D}}) where {T,A,L,B,C,D}
    PVector{ParamVector{T,A,Int(L/num_times(r))},B,C,D,T}
  end
  V = get_vector_type(r.space)
  newV = change_length(V)
  return newV
end

function GridapDistributed.local_views(r::DistributedRBSpace)
  map(
    local_views(r.space),
    local_views(get_basis_space(r)),
    local_views(get_basis_time(r))
    ) do space,basis_space,basis_time

    RBSpace(space,basis_space,basis_time)
  end
end

function RB.compress(s::DistributedTransientSnapshots,r::DistributedRBSpace)
  map(own_values(s),local_views(r)) do s,r
    compress(s,r)
  end
end

function RB.compress(
  s::DistributedTransientSnapshots,
  trial::DistributedRBSpace,
  test::DistributedRBSpace;
  kwargs...)

  map(own_values(s),local_views(trial),local_views(test)) do s,trial,test
    compress(s,trial,test;kwargs...)
  end
end

function RB.recast(red_x::AbstractVector,r::DistributedRBSpace)
  own_vals = map(local_views(red_x),local_views(r)) do red_x,r
    recast(red_x,r)
  end
  x = zero_free_values(r.space)
  map(copy!,own_values(x),own_vals)
  consistent!(x) |> wait
  return x
end

function RB.compress_basis_space(A::PMatrix,test::RBSpace)
  basis_test = get_basis_space(test)
  map(eachcol(A)) do a
    basis_test'*a
  end
end

function RB.compress_basis_space(A::PMatrix,trial::RBSpace,test::RBSpace)
  basis_test = get_basis_space(test)
  basis_trial = get_basis_space(trial)
  map(get_values(A)) do A
    basis_test'*A*basis_trial
  end
end

function RB.combine_basis_time(
  trial::DistributedRBSpace,
  test::DistributedRBSpace;
  kwargs...)

  map(local_views(trial),local_views(test)) do trial,test
    RB.combine_basis_time(trial,test;kwargs...)
  end
end

function RB.mdeim(
  solver::RBSolver,
  fs::DistributedFESpace,
  trian::DistributedTriangulation,
  basis_space::AbstractMatrix,
  basis_time::AbstractMatrix)

  lu_interp,red_trian,integration_domain = map(
    local_views(fs),local_views(trian),local_views(basis_space),local_views(basis_time)
    ) do fs,trian,basis_space,basis_time
    mdeim(solver,fs,trian,basis_space,basis_time)
  end |> tuple_of_arrays
  d_red_trian = DistributedTriangulation(red_trian)
  return lu_interp,d_red_trian,integration_domain
end

# function RB.reduced_residual(
#   solver::RBSolver,
#   op::RBOperator,
#   c::Contribution{DistributedTriangulation})

#   info = RB.get_info(solver)
#   a = distributed_array_contribution()
#   for (trian,values) in c.dict
#     RB.reduced_residual!(a,info,op,values,trian)
#   end
#   return a
# end

# function RB.reduced_jacobian(
#   solver::RBSolver,
#   op::RBOperator,
#   c::Contribution{DistributedTriangulation};
#   kwargs...)

#   info = RB.get_info(solver)
#   a = distributed_array_contribution()
#   for (trian,values) in c.dict
#     RB.reduced_jacobian!(a,info,op,values,trian;kwargs...)
#   end
#   return a
# end
