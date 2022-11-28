function Gridap.Algebra.allocate_matrix(::Type{T}) where T
  Matrix{T}[]
end

function Gridap.Algebra.allocate_vector(::Type{T}) where T
  Vector{T}[]
end

function allocate_snapshot(id::Symbol,empty_snap::AbstractArray{T}) where T
  Snapshot(id,empty_snap)
end
