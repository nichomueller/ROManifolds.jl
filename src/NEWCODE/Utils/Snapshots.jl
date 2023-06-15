function generate_snapshots(feop,solver,nsnaps)
  sols = solve(solver,feop,nsnaps)
  cache = snapshots_cache(feop,solver)
  snaps = pmap(sol->collect_snapshot!(cache,sol),sols)
  mat_snaps = compress(first.(snaps))
  param_snaps = Table(last.(snaps))
  Snapshots(mat_snaps,param_snaps)
end

abstract type Snapshots{T} end

mutable struct SingleFieldSnapshots{T} <: Snapshots{T}
  snaps::NnzMatrix{T}
  params::Table
end

mutable struct MultiFieldSnapshots{T} <: Snapshots{T}
  snaps::Vector{NnzMatrix{T}}
  params::Table
end

Base.length(s::Snapshots) = length(s.params)

Base.size(s::Snapshots,idx...) = size(s.snaps,idx...)

nfields(s::MultiFieldSnapshots) = length(s.snaps)

Gridap.CellData.get_data(s::Snapshots) = s.snaps,s.params

complementary_dimension(s::SingleFieldSnapshots) = Int(size(s.snaps,2)/length(s))

istransient(s::Snapshots) = all(tndofs -> tndofs > 1,complementary_dimension(s))

function Snapshots(snaps::NnzMatrix{T},params::Table) where T
  SingleFieldSnapshots{T}(snaps,params)
end

function Snapshots(snaps::Vector{NnzMatrix{T}},params::Table) where T
  MultiFieldSnapshots{T}(snaps,params)
end

function get_single_field(s::MultiFieldSnapshots,fieldid::Int)
  Snapshots(s.snaps[fieldid],s.params)
end

function collect_single_fields(s::MultiFieldSnapshots)
  map(fieldid -> get_single_field(s,fieldid),1:nfields(s))
end

function snapshots_cache(feop,args...)
  param_cache = realization(feop)
  sol_cache = solution_cache(feop.test,args...)
  sol_cache,param_cache
end

function tpod(s::SingleFieldSnapshots;kwargs...)
  basis_space = copy(s.snaps)
  tpod!(basis_space;kwargs...)
  basis_space
end

function transient_tpod(s::SingleFieldSnapshots;kwargs...)
  compress_rows = _compress_rows(s.snaps)
  transient_tpod(Val{compress_rows}(),s;kwargs...)
end

function transient_tpod(::Val{false},s::SingleFieldSnapshots;kwargs...)
  nparams = length(s)
  basis_space = copy(s.snaps)
  tpod!(basis_space;kwargs...)
  basis_time = basis_space'*s.snaps
  change_mode!(basis_time,nparams)
  tpod!(basis_time;kwargs...)

  basis_space,basis_time
end

function transient_tpod(::Val{true},s::SingleFieldSnapshots;kwargs...)
  nparams = length(s)
  basis_time = copy(s.snaps)
  change_mode!(basis_time,nparams)
  bt = copy(basis_time)
  tpod!(basis_time)
  basis_space = basis_time'*bt
  change_mode!(basis_space,nparams)
  tpod!(basis_space)

  basis_space,basis_time
end
