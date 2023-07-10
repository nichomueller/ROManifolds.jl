abstract type Snapshots{T,A} end
abstract type AbstractSingleFieldSnapshots{T,A} <: Snapshots{T,A} end
abstract type AbstractMultiFieldSnapshots{T,A} <: Snapshots{T,A} end

struct SingleFieldSnapshots{T,A} <: AbstractSingleFieldSnapshots{T,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

struct TransientSingleFieldSnapshots{T,A} <: AbstractSingleFieldSnapshots{T,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

struct MultiFieldSnapshots{T,A} <: AbstractMultiFieldSnapshots{T,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

struct TransientMultiFieldSnapshots{T,A} <: AbstractMultiFieldSnapshots{T,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

get_time_ndofs(::Snapshots) = @abstractmethod

get_time_ndofs(s::TransientSingleFieldSnapshots) = Int(size(s,2)/s.nsnaps)

get_time_ndofs(s::TransientMultiFieldSnapshots) = Int(size(s.snaps[1],2)/s.nsnaps)

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

for Tarr in (:Matrix,:Vector)
  @eval begin
    function Snapshots(
      aff::A,
      snaps::Union{$Tarr{T},Vector{$Tarr{T}}},
      nsnaps::Int;
      kwargs...) where {T,A}

      csnaps = compress(snaps)
      Snapshots(aff,csnaps,nsnaps;kwargs...)
    end

    function Snapshots(
      ::A,
      snaps::Vector{Vector{$Tarr{T}}},
      nsnaps::Int;
      type=EMatrix{Float}) where {T,A}

      csnaps = compress(snaps)
      map(s->convert!(type,s),csnaps)
      MultiFieldSnapshots{$Tarr{T},A}(csnaps,nsnaps)
    end
  end
end

Base.length(s::Snapshots) = s.nsnaps

Base.size(s::Snapshots,idx...) = size(s.snaps,idx...)

function Base.copy(s::SingleFieldSnapshots{T,A}) where {T,A}
  SingleFieldSnapshots{T,A}(copy(s.snaps),copy(s.nsnaps))
end

function Base.copy(s::MultiFieldSnapshots{T,A}) where {T,A}
  MultiFieldSnapshots{T,A}(copy(s.snaps),copy(s.nsnaps))
end

function Base.getindex(s::SingleFieldSnapshots,idx...)
  s_copy = copy(s)
  time_ndofs = get_time_ndofs(s)
  ridx = (first(idx...)-1)*time_ndofs+1:last(idx...)*time_ndofs
  s_copy.snaps = s_copy.snaps[:,ridx]
  s_copy.nsnaps = length(idx...)
  s_copy
end

function Base.getindex(s::MultiFieldSnapshots,idx...)
  s_copy = copy(s)
  time_ndofs = get_time_ndofs(s)
  ridx = (first(idx...)-1)*time_ndofs+1:last(idx...)*time_ndofs
  s_copy.snaps = map(x->getindex(x,:,ridx),s_copy.snaps)
  s_copy.nsnaps = length(idx...)
  s_copy
end

function get_datum(s::SingleFieldSnapshots)
  recast(s.snaps)
end

function get_datum(s::MultiFieldSnapshots)
  map(recast,s.snaps)
end

function allocate_matrix(s::SingleFieldSnapshots,sizes...)
  allocate_matrix(s.snaps,sizes...)
end

get_nfields(s::MultiFieldSnapshots) = length(s.snaps)

function get_single_field(s::MultiFieldSnapshots{T,A},fieldid::Int) where {T,A}
  SingleFieldSnapshots{T,A}(s.snaps[fieldid],s.nsnaps)
end

function collect_single_fields(s::MultiFieldSnapshots)
  map(fieldid -> get_single_field(s,fieldid),1:get_nfields(s))
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

function tpod(s::SingleFieldSnapshots;type=Matrix{Float},kwargs...)
  basis_space = tpod(basis_space;kwargs...)
  convert!(type,basis_space)
  basis_space
end

function transient_tpod(
  s::SingleFieldSnapshots,
  args...;
  type=Matrix{Float},
  kwargs...)

  by_row = _compress_rows(s.snaps)
  basis_space,basis_time = transient_tpod(Val{by_row}(),s;kwargs...)
  convert!(type,basis_space)
  convert!(type,basis_time)
  basis_space,basis_time
end

function transient_tpod(::Val{false},s::SingleFieldSnapshots;kwargs...)
  nsnaps = length(s)
  basis_space = tpod(s.snaps;kwargs...)
  compressed_time_snaps = change_mode(basis_space'*s.snaps,nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)

  basis_space,basis_time
end

function transient_tpod(::Val{true},s::SingleFieldSnapshots;kwargs...)
  nsnaps = length(s)
  time_snaps = change_mode(s.snaps,nsnaps)
  basis_time = tpod(time_snaps)
  compressed_space_snaps = change_mode(basis_time'*time_snaps,nsnaps)
  basis_space = tpod(compressed_space_snaps;kwargs...)

  basis_space,basis_time
end

for (T,A) in zip((:AbstractMatrix,:SparseMatrixCSC),
                 (:TimeAffinity,:ParamTimeAffinity))
  @eval begin
    function transient_tpod(
      s::SingleFieldSnapshots{<:$T,$A},
      solver::ODESolver;
      type=Matrix{Float},
      kwargs...)

      basis_space = tpod(s.snaps;kwargs...)
      time_ndofs = get_time_ndofs(solver)
      basis_time = compress(ones(time_ndofs,1))
      convert!(type,basis_space)
      convert!(type,basis_time)
      basis_space,basis_time
    end
  end
end
