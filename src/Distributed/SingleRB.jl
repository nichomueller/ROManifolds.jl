function FEM.get_L2_norm_matrix(
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    get_L2_norm_matrix(trial,test)
  end
end

function FEM.get_H1_norm_matrix(
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    get_H1_norm_matrix(trial,test)
  end
end

struct DistributedTransientSnapshots{T<:AbstractVector{<:RB.AbstractTransientSnapshots}}
  snaps::T
end

function DistributedSnapshots(snaps::AbstractVector{<:RB.AbstractTransientSnapshots})
  DistributedTransientSnapshots(snaps)
end

GridapDistributed.local_views(s::DistributedTransientSnapshots) = s.snaps

function RB.Snapshots(
  values::AbstractVector{<:PVector{V}},
  initial_values::PVector{V},
  args...) where V

  snaps = map(local_views(initial_values),initial_values.index_partition) do ival_part,iip
    part = part_id(iip)
    vals_part = Vector{V}(undef,length(values))
    for (k,v) in enumerate(values)
      map(local_views(v),v.index_partition) do val,ip
        if part_id(ip) == part
          vals_part[k] = val
        end
      end
    end
    Snapshots(vals_part,ival_part,args...)
  end
  DistributedTransientSnapshots(snaps)
end

function RB.Snapshots(
  values::AbstractVector{<:PVector{V}},
  args...) where V

  item = first(values)
  parts = map(part_id,item.index_partition)
  snaps = map(parts) do part
    vals_part = Vector{V}(undef,length(values))
    for (k,v) in enumerate(values)
      map(local_views(v),v.index_partition) do val,ip
        if part_id(ip) == part
          vals_part[k] = val
        end
      end
    end
    Snapshots(vals_part,args...)
  end
  DistributedSnapshots(snaps)
end

function RB.reduced_basis(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  ϵ = RB.get_tol(info)
  nsnaps_state = RB.num_offline_params(info)
  norm_matrix = RB.get_norm_matrix(info,feop)
  basis_space,basis_time = map(local_views(s)) do s
    reduced_basis(s,norm_matrix;ϵ,nsnaps_state)
  end |> tuple_of_arrays
  return basis_space,basis_time
end
