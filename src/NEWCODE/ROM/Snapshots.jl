abstract type GenericTransientSnapshots end
abstract type TransientSnapshots{T,N,A} <: GenericTransientSnapshots end

struct TransientSingleFieldSnapshots{T,A} <: TransientSnapshots{T,1,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

struct TransientMultiFieldSnapshots{T,N,A} <: TransientSnapshots{T,N,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

function TransientSnapshots(
  ::A,
  snaps::NnzArray{T},
  nsnaps::Int;
  type=EMatrix{Float}) where {T,A}

  csnaps = snaps
  convert!(type,csnaps)
  TransientSingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

function TransientSnapshots(
  ::A,
  snaps::Vector{NnzArray{T}},
  nsnaps::Int;
  type=EMatrix{Float}) where {T,A}

  csnaps = hcat(snaps...)
  convert!(type,csnaps)
  TransientSingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

for Tarr in (:Matrix,:Vector)
  @eval begin
    function TransientSnapshots(
      aff::A,
      snaps::Union{$Tarr{T},Vector{$Tarr{T}}},
      nsnaps::Int;
      kwargs...) where {T,A}

      csnaps = compress(snaps)
      $fun(aff,csnaps,nsnaps;kwargs...)
    end

    function TransientSnapshots(
      ::A,
      snaps::Vector{Vector{$Tarr{T}}},
      nsnaps::Int;
      type=EMatrix{Float}) where {T,A}

      N = length(snaps)
      csnaps = compress(snaps)
      map(s->convert!(type,s),csnaps)
      TransientMultiFieldSnapshots{$Tarr{T},N,A}(csnaps,nsnaps)
    end
  end
end

function Base.getindex(
  s::TransientSingleFieldSnapshots{T,A},
  idx1,
  idx2=:) where {T,A}

  time_ndofs = get_time_ndofs(s)
  _idx = ((first(idx1)-1)*time_ndofs+1:last(idx1)*time_ndofs)[idx2]
  snaps = s.snaps[:,_idx]
  nsnaps = length(idx1)
  TransientSingleFieldSnapshots{T,A}(snaps,nsnaps)
end

function Base.getindex(
  s::TransientMultiFieldSnapshots{T,N,A},
  idx1,
  idx2=:) where {T,N,A}

  time_ndofs = get_time_ndofs(s)
  _idx = ((first(idx1)-1)*time_ndofs+1:last(idx1)*time_ndofs)[idx2]
  snaps = map(x->getindex(x,:,_idx),s.snaps)
  nsnaps = length(idx1)
  TransientMultiFieldSnapshots{T,N,A}(snaps,nsnaps)
end

function Base.iterate(s::TransientSingleFieldSnapshots)
  i = 1
  snap_i = get_datum(s[i])
  return snap_i,i+1
end
function Base.iterate(s::TransientSingleFieldSnapshots,idx::Int)
  if idx > s.nsnaps
    return
  end
  i = idx
  snap_i = get_datum(s[i])
  return snap_i,i+1
end

function Base.iterate(s::TransientMultiFieldSnapshots{T,A}) where {T,A}
  fieldid = 1
  snapid = TransientSingleFieldSnapshots{T,A}(s.snaps[fieldid],s.nsnaps)
  return snapid,fieldid+1
end

function Base.iterate(s::TransientMultiFieldSnapshots{T,A},fieldid::Int) where {T,A}
  if fieldid > length(s.snaps)
    return
  end
  snapid = TransientSingleFieldSnapshots{T,A}(s.snaps[fieldid],s.nsnaps)
  return snapid,fieldid+1
end

get_time_ndofs(s::TransientSingleFieldSnapshots) = Int(size(s.snaps,2)/s.nsnaps)

get_time_ndofs(s::TransientMultiFieldSnapshots) = Int(size(s.snaps[1],2)/s.nsnaps)

get_datum(s::TransientSingleFieldSnapshots) = recast(s.snaps)

get_datum(s::TransientMultiFieldSnapshots) = vcat(map(recast,s.snaps))

function convert!(::Type{T},s::TransientSingleFieldSnapshots) where T
  convert!(T,s.snaps)
  return
end

function convert!(::Type{T},s::TransientMultiFieldSnapshots) where T
  for sf in s.snaps
    convert!(T,sf)
  end
  return
end

function tpod(
  s::TransientSingleFieldSnapshots,
  args...;
  type=Matrix{Float},
  kwargs...)

  by_row = _compress_rows(s.snaps)
  basis_space,basis_time = tpod(Val{by_row}(),s;kwargs...)
  convert!(type,basis_space)
  convert!(type,basis_time)
  basis_space,basis_time
end

function tpod(::Val{false},s::TransientSingleFieldSnapshots;kwargs...)
  basis_space = tpod(s.snaps;kwargs...)
  compressed_time_snaps = change_mode(basis_space'*s.snaps,s.nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)

  basis_space,basis_time
end

function tpod(::Val{true},s::TransientSingleFieldSnapshots;kwargs...)
  time_snaps = change_mode(s.snaps,s.nsnaps)
  basis_time = tpod(time_snaps)
  compressed_space_snaps = change_mode(basis_time'*time_snaps,s.nsnaps)
  basis_space = tpod(compressed_space_snaps;kwargs...)

  basis_space,basis_time
end

for A in (:TimeAffinity,:ParamTimeAffinity)
  @eval begin
    function tpod(
      s::TransientSingleFieldSnapshots{T,$A},
      solver::ODESolver;
      type=Matrix{Float},
      kwargs...) where T

      basis_space = tpod(s.snaps;kwargs...)
      time_ndofs = get_time_ndofs(solver)
      basis_time = compress(ones(time_ndofs,1))
      convert!(type,basis_space)
      convert!(type,basis_time)
      basis_space,basis_time
    end
  end
end
