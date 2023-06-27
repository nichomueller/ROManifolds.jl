abstract type Snapshots{T,A} end

mutable struct SingleFieldSnapshots{T,A} <: Snapshots{T,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

mutable struct MultiFieldSnapshots{T,A} <: Snapshots{T,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

Base.length(s::Snapshots) = s.nsnaps

Base.size(s::Snapshots,idx...) = size(s.snaps,idx...)

get_nfields(s::MultiFieldSnapshots) = length(s.snaps)

Gridap.CellData.get_data(s::Snapshots) = recast(s.snaps)

function Snapshots(snaps::NnzArray{T},::A;kwargs...) where {T,A}
  csnaps = snaps
  nsnaps = size(snaps,2)
  SingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

function Snapshots(snaps::Vector{NnzArray{T}},::A;kwargs...) where {T,A}
  csnaps = compress(snaps;kwargs...)
  nsnaps = length(snaps)
  SingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

function Snapshots(snaps::Vector{Vector{NnzArray{T}}},::A;kwargs...) where {T,A}
  csnaps = compress(snaps;kwargs...)
  nsnaps = length(snaps)
  MultiFieldSnapshots{T,A}(csnaps,nsnaps)
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

for (Top,Tsol) in zip((:ParamFEOperator,:ParamTransientFEOperator),(:FESolver,:ODESolver))
  @eval begin
    function generate_solutions(
      feop::$Top,
      fesolver::$Tsol,
      soldata)

      aff = get_affinity(soldata)
      cache = solution_cache(feop.test,fesolver,aff)
      sols = pmap(d->collect_solution!(cache,d),soldata)
      Snapshots(sols,aff)
    end

    function generate_residuals(
      feop::$Top,
      vecdata)

      aff = get_affinity(vecdata)
      cache = residuals_cache(feop.assem,vecdata,aff)
      ress = pmap(d -> collect_residuals!(cache,feop,d),vecdata)
      Snapshots(ress,aff)
    end

    function generate_jacobians(
      feop::$Top,
      matdata)

      aff = get_affinity(matdata)
      cache = jacobian_cache(feop.assem,matdata,aff)
      jacs = pmap(d -> collect_jacobians!(cache,feop,d),matdata)
      Snapshots(jacs,aff)
    end
  end
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
