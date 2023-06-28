abstract type Snapshots{T,A} end

mutable struct SingleFieldSnapshots{T,A} <: Snapshots{T,A}
  snaps::T
  nsnaps::Int
end

mutable struct MultiFieldSnapshots{T,A} <: Snapshots{T,A}
  snaps::Vector{T}
  nsnaps::Int
end

for (T,Tnnz) in zip((:AbstractVector,:AbstractMatrix),
                    (:AbstractNnzVector,:AbstractNnzMatrix))
  @eval begin
    function Snapshots(s::$T,::A;type=EMatrix{Float}) where A
      snaps = isa(s,AbstractMatrix) ? s : reshape(s,:,1)
      csnaps = convert(type,snaps)
      nsnaps = size(snaps,2)
      SingleFieldSnapshots{AbstractMatrix,A}(csnaps,nsnaps)
    end

    function Snapshots(s::$Tnnz,::A;type=EMatrix{Float}) where A
      snaps = isa(s,NnzMatrix) ? s : reshape(s,:,1)
      csnaps = convert(type,snaps)
      nsnaps = size(snaps,2)
      SingleFieldSnapshots{AbstractNnzMatrix,A}(csnaps,nsnaps)
    end

    function Snapshots(s::Vector{<:$T},::A;type=EMatrix{Float}) where A
      snaps = hcat(s...)
      csnaps = convert(type,snaps)
      nsnaps = length(s)
      SingleFieldSnapshots{AbstractMatrix,A}(csnaps,nsnaps)
    end

    function Snapshots(s::Vector{<:$Tnnz},::A;type=EMatrix{Float}) where A
      snaps = s
      csnaps = convert(type,snaps)
      nsnaps = length(s)
      SingleFieldSnapshots{AbstractNnzMatrix,A}(csnaps,nsnaps)
    end
  end
end

function Snapshots(
  s::Vector{Vector{AbstractMatrix}},
  ::A;
  type=EMatrix{Float}) where A

  nfields = length(first(s))
  csnaps = map(n -> convert(type,hcat(map(sn -> getindex(sn,n),s)...)),1:nfields)
  nsnaps = length(snaps)
  MultiFieldSnapshots{AbstractMatrix,A}(csnaps,nsnaps)
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

Gridap.CellData.get_data(s::SingleFieldSnapshots) = s.snaps

for T in (:AbstractMatrix,:AbstractNnzMatrix)
  @eval begin
    function Gridap.CellData.get_data(s::SingleFieldSnapshots{<:$T,A}) where A
      recast(s.snaps)
    end
  end
end

get_nfields(s::MultiFieldSnapshots) = length(s.snaps)

function get_single_field(s::MultiFieldSnapshots,fieldid::Int)
  Snapshots(s.snaps[fieldid],s.nsnaps)
end

function collect_single_fields(s::MultiFieldSnapshots)
  map(fieldid -> get_single_field(s,fieldid),1:get_nfields(s))
end

Base.getindex(s::MultiFieldSnapshots,i::Int) = get_single_field(s,i)

Base.convert(::Type{Any},s::SingleFieldSnapshots) = s

Base.convert(::Type{Any},s::MultiFieldSnapshots) = s

function Base.convert(::Type{T},s::SingleFieldSnapshots) where T
  s_copy = copy(s)
  s_copy.snaps = convert(T,s_copy.nonzero_val)
  s_copy
end

function Base.convert(::Type{T},s::MultiFieldSnapshots) where T
  s_copy = copy(s)
  s_copy.snaps = [convert(T,sf) for sf in s_copy.snaps]
  s_copy
end

function Gridap.FESpaces.allocate_matrix(s::SingleFieldSnapshots,sizes...)
  allocate_matrix(s.snaps,sizes...)
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

for (T,A) in zip((:AbstractMatrix,:AbstractNnzMatrix),
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
