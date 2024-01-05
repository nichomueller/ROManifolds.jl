struct DistributedSnapshots{T<:AbstractVector{<:Snapshots}}
  snaps::T
end

function RB.Snapshots(snaps::Vector{<:PVector{<:PTArray}})
  s1 = first(snaps)
  parts = map(part_id,s1.index_partition)
  snap_parts = map(parts) do part
    snaps = map(snaps) do si
      local_views(si)[part]
    end
    Snapshots(snaps)
  end
  DistributedSnapshots(snap_parts)
end

struct DistributedRBSpace{T<:AbstractVector{<:RBSpace}}
  rbspace::T
end

function RB.reduced_basis(rbinfo::RBInfo,feop::PTFEOperator,s::DistributedSnapshots)
  rbspace = map(snaps.snaps) do s
    reduced_basis(rbinfo,feop,s)
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
