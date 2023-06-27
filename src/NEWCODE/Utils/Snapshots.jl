abstract type Snapshots{T} end

mutable struct SingleFieldSnapshots{T} <: Snapshots{T}
  snaps::NnzArray{T}
  nsnaps::Int
end

mutable struct MultiFieldSnapshots{T} <: Snapshots{T}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

Base.length(s::Snapshots) = s.nsnaps

Base.size(s::Snapshots,idx...) = size(s.snaps,idx...)

get_nfields(s::MultiFieldSnapshots) = length(s.snaps)

Gridap.CellData.get_data(s::Snapshots) = recast(s.snaps)

function Snapshots(csnaps::NnzArray{T},nsnaps=size(snaps,2);kwargs...) where T
  SingleFieldSnapshots{T}(csnaps,nsnaps)
end

function Snapshots(snaps::Vector{NnzArray{T}};kwargs...) where T
  csnaps = compress(snaps;kwargs...)
  nsnaps = length(snaps)
  SingleFieldSnapshots{T}(csnaps,nsnaps)
end

function Snapshots(snaps::Vector{Vector{NnzArray{T}}};kwargs...) where T
  csnaps = compress(snaps;kwargs...)
  nsnaps = length(snaps)
  MultiFieldSnapshots{T}(csnaps,nsnaps)
end

function get_single_field(s::MultiFieldSnapshots,fieldid::Int)
  Snapshots(s.snaps[fieldid],s.nsnaps)
end

function collect_single_fields(s::MultiFieldSnapshots)
  map(fieldid -> get_single_field(s,fieldid),1:get_nfields(s))
end

Base.getindex(s::MultiFieldSnapshots,i::Int) = get_single_field(s,i)

function convert!(::Type{T},s::SingleFieldSnapshots) where T
  convert!(T,s.snaps)
  s
end

function convert!(::Type{T},s::MultiFieldSnapshots) where T
  for sf in s.snaps
    convert!(T,sf)
  end
  sf
end

function tpod(s::SingleFieldSnapshots;type=Matrix{Float},kwargs...)
  basis_space = copy(s.snaps)
  tpod!(basis_space;kwargs...)
  convert!(type,basis_space)
end

function transient_tpod(s::SingleFieldSnapshots;type=Matrix{Float},kwargs...)
  compress_rows = _compress_rows(s.snaps)
  basis_space,basis_time = transient_tpod(Val{compress_rows}(),s;kwargs...)
  convert!(type,basis_space),convert!(type,basis_time)
end

function transient_tpod(::Val{false},s::SingleFieldSnapshots;kwargs...)
  nsnaps = length(s)
  basis_space = copy(s.snaps)
  tpod!(basis_space;kwargs...)
  basis_time = basis_space'*s.snaps
  change_mode!(basis_time,nsnaps)
  tpod!(basis_time;kwargs...)

  basis_space,basis_time
end

function transient_tpod(::Val{true},s::SingleFieldSnapshots;kwargs...)
  nsnaps = length(s)
  basis_time = copy(s.snaps)
  change_mode!(basis_time,nsnaps)
  bt = copy(basis_time)
  tpod!(basis_time)
  basis_space = basis_time'*bt
  change_mode!(basis_space,nsnaps)
  tpod!(basis_space)

  basis_space,basis_time
end
