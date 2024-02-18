struct DistributedTransientSnapshots{T}
  snaps::T
end

DistributedSnapshots(snaps) = DistributedTransientSnapshots(snaps)

Base.axes(s::DistributedTransientSnapshots,i...) = axes(s.snaps,i...)

GridapDistributed.local_views(s::DistributedTransientSnapshots) = local_values(s.snaps)

function PartitionedArrays.partition(s::DistributedTransientSnapshots)
  partition(s.snaps)
end

function PartitionedArrays.local_values(s::DistributedTransientSnapshots)
  local_values(s.snaps)
end

function PartitionedArrays.own_values(s::DistributedTransientSnapshots)
  map(partition(s),partition(axes(s,1))) do values,indices_rows
    select_snapshots(values,spacerange=own_to_local(indices_rows))
  end
end

function PartitionedArrays.ghost_values(s::DistributedTransientSnapshots)
  map(partition(s),partition(axes(s,1))) do values,indices_rows
    select_snapshots(values,spacerange=ghost_to_local(indices_rows))
  end
end

function RB.Snapshots(
  values::PVector{P},
  args...) where {P<:AbstractParamContainer}

  index_partition = values.index_partition
  snaps = map(local_views(values)) do values
    Snapshots(values,args...)
  end
  psnaps = PMatrix(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.Snapshots(
  values::AbstractVector{<:PVector{P}},
  args...) where {P<:AbstractParamContainer}

  index_partition = first(values).index_partition
  parts = map(part_id,index_partition)
  snaps = map(parts) do part
    vals_part = Vector{P}(undef,length(values))
    for (k,v) in enumerate(values)
      map(local_views(v),index_partition) do val,ip
        if part_id(ip) == part
          vals_part[k] = val
        end
      end
    end
    Snapshots(vals_part,args...)
  end
  psnaps = PMatrix(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.Snapshots(a::DistributedArrayContribution,args...)
  b = Contribution(IdDict{DistributedTriangulation,DistributedTransientSnapshots}())
  for (trian,values) in a.dict
    b[trian] = Snapshots(values,args...)
  end
  b
end

function RB.get_realization(s::DistributedTransientSnapshots)
  s1 = PartitionedArrays.getany(local_views(s.snaps))
  get_realization(s1)
end

function RB.get_values(s::DistributedTransientSnapshots)
  snaps = map(local_views(s)) do s
    get_values(s)
  end
  row_partition = s.snaps.row_partition
  PVector(snaps,row_partition)
end

function RB.select_snapshots(s::DistributedTransientSnapshots,args...;kwargs...)
  snaps = map(local_views(s)) do s
    select_snapshots(s,args...;kwargs...)
  end
  row_partition = s.snaps.row_partition
  psnaps = PMatrix(snaps,row_partition)
  DistributedSnapshots(psnaps)
end

function RB.reverse_snapshots(s::DistributedTransientSnapshots)
  snaps = map(local_views(s)) do s
    reverse_snapshots(s)
  end
  row_partition = s.snaps.row_partition
  psnaps = PMatrix(snaps,row_partition)
  DistributedSnapshots(psnaps)
end

const DistributedTransientNnzSnapshots = DistributedTransientSnapshots{T} where {
  T<:Union{<:PSparseMatrix,AbstractVector{<:PSparseMatrix}}}

function RB.Snapshots(
  values::PSparseMatrix{P},
  args...) where {P<:AbstractParamContainer}

  row_partition = values.row_partition
  col_partition = values.col_partition
  snaps = map(local_views(values)) do values
    Snapshots(values,args...)
  end
  psnaps = PSparseMatrix(snaps,row_partition,col_partition)
  DistributedSnapshots(psnaps)
end

function PartitionedArrays.own_values(s::DistributedTransientNnzSnapshots{<:PSparseMatrix})
  map(partition(s),partition(axes(s,1)),partition(axes(s,2))) do v,indices_rows,indices_cols
    ovals = own_values(v.values,indices_rows,indices_cols)
    Snapshots(ovals,get_realization(s))
  end
end

function PartitionedArrays.ghost_values(s::DistributedTransientNnzSnapshots{<:PSparseMatrix})
  map(partition(s),partition(axes(s,1)),partition(axes(s,2))) do v,indices_rows,indices_cols
    gvals = ghost_values(v.values,indices_rows,indices_cols)
    Snapshots(gvals,get_realization(s))
  end
end

function PartitionedArrays.own_ghost_values(s::DistributedTransientNnzSnapshots{<:PSparseMatrix})
  map(partition(s),partition(axes(s,1)),partition(axes(s,2))) do v,indices_rows,indices_cols
    gvals = own_ghost_values(v.values,indices_rows,indices_cols)
    Snapshots(gvals,get_realization(s))
  end
end

function PartitionedArrays.ghost_own_values(s::DistributedTransientNnzSnapshots{<:PSparseMatrix})
  map(partition(s),partition(axes(s,1)),partition(axes(s,2))) do v,indices_rows,indices_cols
    gvals = ghost_own_values(v.values,indices_rows,indices_cols)
    Snapshots(gvals,get_realization(s))
  end
end

function RB.get_values(s::DistributedTransientNnzSnapshots)
  snaps = map(local_views(s)) do s
    get_values(s)
  end
  row_partition = s.snaps.row_partition
  col_partition = s.snaps.col_partition
  PSparseMatrix(snaps,row_partition,col_partition)
end
