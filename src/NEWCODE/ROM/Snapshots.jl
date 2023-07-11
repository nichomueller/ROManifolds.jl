abstract type Snapshots{T,N,A} end
abstract type TransientSnapshots{T,N,A} end

struct SingleFieldSnapshots{T,A} <: Snapshots{T,1,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

struct MultiFieldSnapshots{T,N,A} <: Snapshots{T,N,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

struct TransientSingleFieldSnapshots{T,A} <: TransientSnapshots{T,1,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

struct TransientMultiFieldSnapshots{T,N,A} <: TransientSnapshots{T,N,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

function Snapshots(
  ::A,
  snaps::NnzArray{T},
  nsnaps::Int;
  type=EMatrix{Float}) where {T,A}

  csnaps = snaps
  convert!(type,csnaps)
  SingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

function Snapshots(
  ::A,
  snaps::Vector{NnzArray{T}},
  nsnaps::Int;
  type=EMatrix{Float}) where {T,A}

  csnaps = hcat(snaps)
  convert!(type,csnaps)
  SingleFieldSnapshots{T,A}(csnaps,nsnaps)
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

  csnaps = hcat(snaps)
  convert!(type,csnaps)
  TransientSingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

for (fun,Tarr) in zip((:Snapshots,:TransientSnapshots),(:Matrix,:Vector))
  @eval begin
    function $fun(
      aff::A,
      snaps::Union{$Tarr{T},Vector{$Tarr{T}}},
      nsnaps::Int;
      kwargs...) where {T,A}

      csnaps = compress(snaps)
      $fun(aff,csnaps,nsnaps;kwargs...)
    end

    function Snapshots(
      ::A,
      snaps::Vector{Vector{$Tarr{T}}},
      nsnaps::Int;
      type=EMatrix{Float}) where {T,A}

      N = length(snaps)
      csnaps = compress(snaps)
      map(s->convert!(type,s),csnaps)
      MultiFieldSnapshots{$Tarr{T},N,A}(csnaps,nsnaps)
    end

    function TransientSnapshots(
      ::A,
      snaps::Vector{Vector{$Tarr{T}}},
      nsnaps::Int;
      type=EMatrix{Float}) where {T,A}

      N = length(snaps)
      csnaps = compress(snaps)
      map(s->convert!(type,s),csnaps)
      TransientMultiSnapshots{$Tarr{T},N,A}(csnaps,nsnaps)
    end
  end
end

function Base.copy(s::SingleFieldSnapshots{T,A}) where {T,A}
  SingleFieldSnapshots{T,A}(copy(s.snaps),copy(s.nsnaps))
end

function Base.copy(s::MultiFieldSnapshots{T,N,A}) where {T,N,A}
  MultiFieldSnapshots{T,N,A}(copy(s.snaps),copy(s.nsnaps))
end

function Base.copy(s::TransientSingleFieldSnapshots{T,A}) where {T,A}
  TransientSingleFieldSnapshots{T,A}(copy(s.snaps),copy(s.nsnaps))
end

function Base.copy(s::TransientMultiFieldSnapshots{T,N,A}) where {T,N,A}
  TransientMultiFieldSnapshots{T,N,A}(copy(s.snaps),copy(s.nsnaps))
end

function Base.getindex(s::SingleFieldSnapshots,idx)
  s_copy = copy(s)
  _idx = first(idx):last(idx)
  s_copy.snaps = s_copy.snaps[:,_idx]
  s_copy.nsnaps = length(idx)
  s_copy
end

function Base.getindex(s::MultiFieldSnapshots,idx)
  s_copy = copy(s)
  _idx = first(idx):last(idx)
  s_copy.snaps = map(x->getindex(x,:,_idx),s_copy.snaps)
  s_copy.nsnaps = length(idx)
  s_copy
end

function Base.getindex(s::TransientSingleFieldSnapshots,idx1,idx2)
  s_copy = copy(s)
  time_ndofs = get_time_ndofs(s)
  _idx = ((first(idx1)-1)*time_ndofs+1:last(idx1)*time_ndofs)[idx2]
  s_copy.snaps = s_copy.snaps[:,_idx]
  s_copy.nsnaps = length(idx...)
  s_copy
end

function Base.getindex(s::TransientMultiFieldSnapshots,idx1,idx2)
  s_copy = copy(s)
  time_ndofs = get_time_ndofs(s)
  _idx = ((first(idx1)-1)*time_ndofs+1:last(idx1)*time_ndofs)[idx2]
  s_copy.snaps = map(x->getindex(x,:,_idx),s_copy.snaps)
  s_copy.nsnaps = length(idx...)
  s_copy
end

for Tsnp in (:MultiFieldSnapshots,:TransientMultiFieldSnapshots)
  @eval begin
    function Base.length(s::$Tsnp)
      length(s.snaps)
    end

    function Base.iterate(s::$Tsnp,idx::Int)
      iterate(s.snaps,idx)
    end
  end
end

get_time_ndofs(::Snapshots) = @abstractmethod

get_time_ndofs(s::TransientSingleFieldSnapshots) = Int(size(s,2)/s.nsnaps)

get_time_ndofs(s::TransientMultiFieldSnapshots) = Int(size(s.snaps[1],2)/s.nsnaps)

get_datum(s::SingleFieldSnapshots) = recast(s.snaps)

get_datum(s::MultiFieldSnapshots) = map(recast,s.snaps)

get_datum(s::TransientSingleFieldSnapshots) = recast(s.snaps)

get_datum(s::TransientMultiFieldSnapshots) = map(recast,s.snaps)

function allocate_matrix(s::SingleFieldSnapshots,sizes...)
  allocate_matrix(s.snaps,sizes...)
end

function convert!(::Type{T},s::SingleFieldSnapshots) where T
  convert!(T,s.snaps)
  return
end

function convert!(::Type{T},s::MultiFieldSnapshots) where T
  for sf in s.snaps
    convert!(T,sf)
  end
  return
end

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

function tpod(s::SingleFieldSnapshots;type=Matrix{Float},kwargs...)
  basis_space = tpod(basis_space;kwargs...)
  convert!(type,basis_space)
  basis_space
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
