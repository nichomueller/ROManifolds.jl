struct Snapshots{T,A}
  snaps::AbstractArray
  nsnaps::Int
  function Snapshots(::A,snaps::AbstractArray,nsnaps::Int) where A
    T = eltype(snaps)
    new{T,A}(snaps,nsnaps)
  end
end

const SingleFieldSnapshots{T,A} = Snapshots{<:AbstractArray{T},A}
const MultiFieldSnapshots{T,A} = Snapshots{<:BlockArray{T},A}

function Base.collect(snap::Snapshots)
  lazy_snaps = get_snaps(snap)
  collect(lazy_snaps)
end

function Base.size(snap::Snapshots)
  len = snap.nsnaps
  s1 = first(snap.snaps)
  size(s1,1),size(s1,2)*len
end

function Base.show(io::IO,snap::Snapshots{T,A}) where {T,A}
  print(io,"Structure storing $(snap.nsnaps) $A snapshots of eltype $T")
end

get_snaps(snap::Snapshots) = snap.snaps

get_nsnaps(snap::Snapshots) = snap.nsnaps

get_time_ndofs(snap::Snapshots) = size(first(snap.snaps),2)

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
    function tpod(snap::Snapshots{$A},args...;kwargs...)
      snaps = collect(snap)
      nsnaps = get_nsnaps(snap)
      time_ndofs = Int(size(snaps,2)/nsnaps)
      T = eltype(snaps)

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
  nsnaps=50)

  aff = NonAffinity()
  nsols = nsnaps
  printstyled("Generating $nsols solution snapshots\n";color=:blue)

  sols = collect_solutions(fesolver,feop,params)
  Snapshots(aff,sols,nsols)
end

get_nsnaps(::Affinity,nsnaps) = nsnaps
get_nsnaps(::Union{TimeAffinity,NonAffinity},nsnaps) = 1

function collect_residuals(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  params::Table,
  trian::Triangulation,
  args...;
  nsnaps=50)

  aff = affinity_residual(feop,fesolver,trian)
  nress = get_nsnaps(aff,nsnaps)

  printstyled("Generating $nress residuals snapshots\n";color=:blue)
  sols = view(get_snaps(snaps),1:nress)
  params = view(params,1:nress)
  ress = collect_residuals(feop,fesolver,sols,params,trian,args...)

  Snapshots(aff,ress,nress)
end

function collect_jacobians(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  params::Table,
  trian::Triangulation,
  args...;
  i=1,nsnaps=50)

  aff = affinity_jacobian(feop,fesolver,trian)
  njacs = get_nsnaps(aff,nsnaps)

  printstyled("Generating $njacs jacobians snapshots\n";color=:blue)
  sols = view(get_snaps(snaps),1:njacs)
  params = view(params,1:njacs)
  jacs = collect_jacobians(feop,fesolver,sols,params,trian,args...;i)

  Snapshots(collector,jacs,njacs)
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
