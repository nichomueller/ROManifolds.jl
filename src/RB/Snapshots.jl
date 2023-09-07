struct Snapshots{T,A}
  snaps::LazyArray
  function Snapshots(::A,snaps::LazyArray) where A
    T = eltype(snaps)
    new{T,A}(snaps)
  end
end

const SingleFieldSnapshots{T,A} = Snapshots{<:AbstractArray{T},A}
const MultiFieldSnapshots{T,A} = Snapshots{<:BlockArray{T},A}

Base.length(snap::Snapshots) = length(snap.snaps)

function Base.size(snap::Snapshots)
  lazy_snaps = get_snaps(snap)
  nsnaps = length(snap)
  cache = array_cache(lazy_snaps)
  s1 = getindex!(cache,lazy_snaps,1)
  size(s1,1),size(s1,2)*nsnaps
end

function Base.collect(snap::Snapshots)
  lazy_snaps = get_snaps(snap)
  nsnaps = length(snap)
  cache = array_cache(lazy_snaps)
  T = eltype(lazy_snaps)
  snaps = Vector{T}(undef,nsnaps)
  for i = 1:nsnaps
    snaps[i] = getindex!(cache,lazy_snaps,i)
  end
  return snaps
end

function Base.show(io::IO,snap::Snapshots{T,A}) where {T,A}
  nsnaps = length(snap)
  print(io,"Structure storing $nsnaps $A snapshots of eltype $T")
end

get_snaps(snap::Snapshots) = snap.snaps

function tpod(snap::Snapshots,args...;kwargs...)
  s = size(snap)
  tpod(Val(s[1] < s[2]),snap,args...;kwargs...)
end

function tpod(::Val{true},snap::Snapshots,args...;kwargs...)
  snaps = collect(snap)
  nsnaps = get_nsnaps(snap)

  basis_space = tpod(snaps,args...;kwargs...)
  compressed_space_snaps = prod(basis_space,snaps)
  compressed_time_snaps = change_mode(compressed_space_snaps,nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)
  basis_space,basis_time
end

function Arrays.return_cache(::typeof(transpose),a::AbstractArray)
  CachedArray(a)
end

function Arrays.evaluate!(cache,::typeof(transpose),a::AbstractArray)
  s = size(a)
  st = filter(!isempty,(s[1:end-2],s[end],s[end-1]))
  setsize!(cache,st)
  b = cache.array
  b = transpose(a)
  b
end

function tpod(::Val{false},snap::Snapshots,args...;kwargs...)
  snaps_t = collect(lazy_map(transpose,get_snaps(snap)))
  nsnaps = get_nsnaps(snap)

  basis_time = tpod(snaps_t;kwargs...)
  compressed_time_snaps = prod(basis_time,snaps_t)
  compressed_space_snaps = change_mode(compressed_time_snaps,nsnaps)
  basis_space = tpod(compressed_space_snaps,args...;kwargs...)
  basis_space,basis_time
end

for A in (:TimeAffinity,:ParamTimeAffinity)
  @eval begin
    function tpod(snap::Snapshots{T,$A},args...;kwargs...) where T
      snaps = collect(snap)
      nsnaps = get_nsnaps(snap)
      time_ndofs = Int(size(snaps,2)/nsnaps)
      S = eltype(T)

      basis_space = tpod(snaps,args...;kwargs...)
      basis_time = ones(S,time_ndofs,1)
      basis_space,basis_time
    end
  end
end

function collect_solutions(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  params::Table;
  nsnaps=50)

  aff = NonAffinity()
  nsols = nsnaps
  printstyled("Generating $nsols solution snapshots\n";color=:blue)

  sols = collect_solutions(fesolver,feop,params)
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
