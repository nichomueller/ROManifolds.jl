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

struct DistributedSnapshots{T<:AbstractVector{<:Snapshots}}
  snaps::T
end

function RB.Snapshots(snaps::Vector{<:PVector{<:PTArray}})
  s1 = first(snaps)
  parts = map(part_id,s1.index_partition)
  snap_parts = map(parts) do part
    snap_part = map(snaps) do si
      local_views(si)[part]
    end
    Snapshots(snap_part)
  end
  DistributedSnapshots(snap_parts)
end

function Base.getindex(s::DistributedSnapshots,idx)
  map(s.snaps) do snaps
    getindex(snaps,idx)
  end
end

struct DistributedRBSpace{T<:AbstractVector{<:RBSpace}}
  rbspace::T
end

function RB.reduced_basis(rbinfo::RBInfo,feop::PTFEOperator,s::DistributedSnapshots)
  rbspace = map(s.snaps) do snaps
    reduced_basis(rbinfo,feop,snaps)
  end
  DistributedRBSpace(rbspace)
end

function project_recast(snap::PVector{<:PTArray},rb::DistributedRBSpace)
  map(local_views(snap),rb.rbspace) do s,rb
    mat = stack(s.array)
    rb_proj = space_time_projection(mat,rb)
    recast(rb_proj,rb)
  end
end
