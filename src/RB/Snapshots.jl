struct Snapshots{T,A}
  snaps::AbstractVector{<:NnzArray{T,2}}
  function Snapshots(::A,snaps::AbstractVector{<:AbstractArray{T}}) where {A,T}
    snnz = map(compress,snaps)
    new{T,A}(snnz)
  end
end

const SingleFieldSnapshots{T,A} = Snapshots{T,A} where {T<:AbstractArray}
const MultiFieldSnapshots{T,A} = Snapshots{T,A} where {T<:BlockArray}

Base.length(snap::Snapshots) = length(snap.snaps)

function Base.size(snap::Snapshots)
  snaps = get_snaps(snap)
  nsnaps = length(snap)
  s1 = first(snaps)
  size(s1,1),size(s1,2)*nsnaps
end

Base.getindex(snap::Snapshots,idx) = getindex(snap.snaps,idx)

Base.collect(snap::Snapshots;transpose=false) = _collect_snaps(Val(transpose),snap)

function _collect_snaps(::Val{false},snap::Snapshots{T}) where T
  snaps = get_snaps(snap)
  return hcat(snaps...)
end

function _collect_snaps(::Val{true},snap::Snapshots{T}) where T
  snaps = get_snaps(snap)
  snaps_t = map(transpose,snaps)
  return hcat(snaps_t...)
end

function Base.show(io::IO,snap::Snapshots{T,A}) where {T,A}
  nsnaps = length(snap)
  print(io,"Structure storing $nsnaps $A snapshots of eltype $T")
end

get_snaps(snap::Snapshots) = snap.snaps

function compress_snapshots(snap::Snapshots,args...;kwargs...)
  s = size(snap)
  if s[1] < s[2]
    snaps = collect(snap)
  else
    snaps = collect(snap;transpose=true)
  end
  nsnaps = get_nsnaps(snap)
  b1 = tpod(snaps,args...;kwargs...)
  compressed_b1 = b1'*snaps
  compressed_snaps_t = change_mode(compressed_b1,nsnaps)
  b2 = tpod(compressed_snaps_t;kwargs...)
  if s[1] < s[2]
    return b1,b2
  else
    return b2,b1
  end
end

for A in (:TimeAffinity,:ParamTimeAffinity)
  @eval begin
    function compress_snapshots(snap::Snapshots{T,$A},args...;kwargs...) where T
      snaps = collect(snap)
      nsnaps = get_nsnaps(snap)
      time_ndofs = Int(size(snaps)[2]/nsnaps)

      basis_space = tpod(snaps,args...;kwargs...)
      basis_time = ones(T,time_ndofs,1)
      basis_space,basis_time
    end
  end
end

function collect_solutions(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  params::Table;
  nsnaps=length(params))

  aff = NonAffinity()
  nsols = nsnaps
  printstyled("Generating $nsols solution snapshots\n";color=:blue)

  sols = solve(fesolver,feop,params)
  Snapshots(aff,sols)
end

function collect_residuals(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  params::Table,
  trian::Triangulation,
  args...;
  kwargs...)

  printstyled("Generating residuals snapshots\n";color=:blue)
  sols = get_snaps(snaps)
  aff,ress = collect_residuals(feop,fesolver,sols,params,trian,args...;kwargs...)

  Snapshots(aff,ress)
end

function collect_jacobians(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  params::Table,
  trian::Triangulation,
  args...;
  kwargs...)

  printstyled("Generating jacobians snapshots\n";color=:blue)
  sols = get_snaps(snaps)
  aff,jacs = collect_jacobians(feop,fesolver,sols,params,trian,args...;kwargs...)

  Snapshots(aff,jacs)
end

function save(info::RBInfo,snap::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps")
    save(path,snap)
  end
end

function save(info::RBInfo,params::Table)
  if info.save_structures
    path = joinpath(info.fe_path,"params")
    save(path,params)
  end
end

function load(T::Type{Snapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps")
  snap = load(T,path)
  snap
end

function load(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params")
  load(T,path)
end
