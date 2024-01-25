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

function RB.Snapshots(snaps::Vector{<:PVector{<:ParamArray}})
  _type(a::PVector{V}) where V = V
  # snap_parts = map(parts) do part
  #   snap_part = map(snaps) do si
  #     local_views(si)[part]
  #   end
  #   Snapshots(snap_part)
  # end
  s1 = first(snaps)
  T = _type(s1)
  parts = map(part_id,s1.index_partition)
  snap_parts = map(parts) do part
    cache = T[]
    for si in snaps
      map(local_views(si),si.index_partition) do sij,j
        if j == part
          push!(cache,sij)
        end
        cache
      end
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

function RB.reduced_basis(rbinfo::RBInfo,feop::TransientParamFEOperator,s::DistributedSnapshots)
  rbspace = map(s.snaps) do snaps
    reduced_basis(rbinfo,feop,snaps)
  end
  DistributedRBSpace(rbspace)
end

function RB.project_recast(snap::AbstractVector{<:ParamArray},rb::DistributedRBSpace)
  map(snap,rb.rbspace) do s,rb
    mat = stack(s.array)
    rb_proj = space_time_projection(mat,rb)
    array = recast(rb_proj,rb)
    ParamArray(array)
  end
end
