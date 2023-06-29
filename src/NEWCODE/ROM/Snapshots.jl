abstract type Snapshots{T,A} end

mutable struct SingleFieldSnapshots{T,A} <: Snapshots{T,A}
  snaps::NnzArray{T}
  nsnaps::Int
end

mutable struct MultiFieldSnapshots{T,A} <: Snapshots{T,A}
  snaps::Vector{NnzArray{T}}
  nsnaps::Int
end

function Snapshots(
  snaps::NnzArray{T},
  ::A;
  type=EMatrix{Float}) where {T,A}

  csnaps = snaps
  convert!(type,csnaps)
  nsnaps = size(snaps,2)
  SingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

function Snapshots(
  snaps::Vector{NnzArray{T}},
  ::A;
  type=EMatrix{Float}) where {T,A}

  csnaps = hcat(snaps)
  convert!(type,csnaps)
  nsnaps = length(snaps)
  SingleFieldSnapshots{T,A}(csnaps,nsnaps)
end

function Snapshots(
  snaps::Vector{Vector{NnzArray{T}}},
  ::A;
  type=EMatrix{Float}) where {T,A}

  nfields = length(first(s))
  csnaps = map(n->hcat(map(sn -> getindex(sn,n),snaps)...),1:nfields)
  convert!(type,csnaps)
  nsnaps = length(snaps)
  MultiFieldSnapshots{T,A}(csnaps,nsnaps)
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
  ridx = (first(idx...)-1)*s_copy.nsnaps+1:last(idx...)*s_copy.nsnaps
  s_copy.snaps = s_copy.snaps[:,ridx]
  s_copy.nsnaps = length(idx...)
  s_copy
end

Base.getindex(s::MultiFieldSnapshots,i::Int) = get_single_field(s,i)

function Gridap.CellData.get_data(s::SingleFieldSnapshots)
  recast(s.snaps)
end

function Gridap.FESpaces.allocate_matrix(s::SingleFieldSnapshots,sizes...)
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

for (Top,Tslv) in zip((:ParamFEOperator,:ParamTransientFEOperator),(:FESolver,:ODESolver))
  @eval begin
    function generate_solutions(
      feop::$Top,
      fesolver::$Tslv,
      params::Table)

      cache = solution_cache(feop.test,fesolver)
      sols = pmap(p->collect_solution!(cache,feop,fesolver,p),params)
      Snapshots(sols,NonAffinity())
    end

    function generate_residuals(
      feop::$Top,
      fesolver::$Tslv,
      params::Table,
      vecdata)

      aff = get_affinity(fesolver,params,vecdata)
      data = get_data(aff,fesolver,params,vecdata)
      cache = residuals_cache(feop.assem,data)
      ress = pmap(d -> collect_residuals!(cache,feop,d),data)
      Snapshots(ress,aff)
    end

    function generate_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      params::Table,
      matdata)

      aff = get_affinity(fesolver,params,matdata)
      data = get_data(aff,fesolver,params,matdata)
      cache = jacobian_cache(feop.assem,data)
      jacs = pmap(d -> collect_jacobians!(cache,feop,d),data)
      Snapshots(jacs,aff)
    end
  end
end

function tpod(s::SingleFieldSnapshots;type=Matrix{Float},kwargs...)
  basis_space = tpod(basis_space;kwargs...)
  convert(type,basis_space)
end

function transient_tpod(
  s::SingleFieldSnapshots,
  args...;
  type=Matrix{Float},
  kwargs...)

  compress_rows = _compress_rows(s.snaps)
  basis_space,basis_time = transient_tpod(Val{compress_rows}(),s;kwargs...)
  convert(type,basis_space),convert(type,basis_time)
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
      basis_time = allocate_matrix(s,time_ndofs,1)
      convert(type,basis_space),convert(type,basis_time)
    end
  end
end
