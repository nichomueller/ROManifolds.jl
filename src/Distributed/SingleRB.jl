function FEM.get_L2_norm_matrix(
  info::RBInfo,
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    get_L2_norm_matrix(info,trial,test)
  end
end

function FEM.get_H1_norm_matrix(
  info::RBInfo,
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    get_H1_norm_matrix(info,trial,test)
  end
end

struct DistributedTransientSnapshots{T<:AbstractVector{<:AbstractTransientSnapshots}}
  snaps::T
end

function DistributedSnapshots(snaps::AbstractVector{<:AbstractTransientSnapshots})
  DistributedTransientSnapshots(snaps)
end

GridapDistributed.local_views(s::DistributedTransientSnapshots) = s.snaps

function RB.Snapshots(
  values::AbstractVector{<:PVector{V}},
  args...) where V

  index_partition = first(values).index_partition
  parts = map(part_id,index_partition)
  snaps = map(parts) do part
    vals_part = Vector{V}(undef,length(values))
    for (k,v) in enumerate(values)
      vector_partition = map(local_views(v),index_partition) do val,ip
        if part_id(ip) == part
          vals_part[k] = val
        end
      end
      PVector(vector_partition,index_partition)
    end
    Snapshots(vals_part,args...)
  end
  DistributedSnapshots(snaps)
end

struct DistributedRBSpace{T<:AbstractVector{<:RBSpace}}
  spaces::T
end

GridapDistributed.local_views(a::DistributedRBSpace) = a.spaces

function RB.reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  ϵ = get_tol(info)
  norm_matrix = get_norm_matrix(info,feop)
  reduced_trial,reduced_test = map(
    local_views(trial),
    local_views(test),
    local_views(s),
    local_views(norm_matrix)
    ) do trial,test,s,norm_matrix

    soff = select_snapshots(s,offline_params(info))
    basis_space,basis_time = reduced_basis(soff,norm_matrix;ϵ)
    reduced_trial = TrialRBSpace(trial,basis_space,basis_time)
    reduced_test = TestRBSpace(test,basis_space,basis_time)
    reduced_trial,reduced_test
  end |> tuple_of_arrays

  dtrial = DistributedRBSpace(reduced_trial)
  dtest = DistributedRBSpace(reduced_test)
  return dtrial,dtest
end

function GridapDistributed._find_vector_type(
  spaces::AbstractVector{<:RBSpace},gids)
  T = get_vector_type(PartitionedArrays.getany(spaces))
  if isa(gids,PRange)
    vector_type = typeof(PVector{T}(undef,partition(gids)))
  else
    vector_type = typeof(BlockPVector{T}(undef,gids))
  end
  return vector_type
end

function RB.compress(r::DistributedRBSpace,xmat::ParamVector{<:AbstractMatrix})
  partition = xmat.index_partition
  vector_partition = map(local_views(r),local_views(xmat)) do r,xmat
    compress(r,xmat)
  end
  PVector(vector_partition,partition)
end

function RB.compress(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  xmat::ParamVector{<:AbstractMatrix};
  kwargs...)

  partition = xmat.index_partition
  vector_partition = map(local_views(trial),local_views(test),local_views(xmat)
    ) do trial,test,xmat
    compress(trial,test,xmat;kwargs...)
  end
  PVector(vector_partition,partition)
end

function recast(r::RBSpace,red_x::ParamVector{<:AbstractMatrix})
  partition = red_x.index_partition
  vector_partition = map(red_x) do red_x
    recast(r,red_x)
  end
  PVector(vector_partition,partition)
end
