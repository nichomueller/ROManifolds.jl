struct Snapshots{T,A}
  snaps::LazyArray
  nsnaps::Int
  function Snapshots(::CollectorMap{A},snaps::LazyArray,nsnaps::Int) where A
    T = eltype(eltype(snaps))
    new{T,A}(snaps,nsnaps)
  end
end

const SingleFieldSnapshots{A} = Snapshots{<:Vector,A}
const MultiFieldSnapshots{A} = Snapshots{<:BlockVector,A}

function Base.collect(snap::Snapshots)
  param_time_snaps = get_snaps(snap)
  param_snaps = lazy_map(x -> reduce(hcat,x),param_time_snaps)
  snaps = collect(param_snaps)
  snaps
end

Base.size(snap::Snapshots,idx...) = size(collect(snap))

get_snaps(snap::Snapshots) = snap.snaps

get_nsnaps(snap::Snapshots) = snap.nsnaps

get_time_ndofs(snap::Snapshots) = Int(size(snap.snaps,2)/snap.nsnaps)

function tpod(snap::Snapshots;kwargs...)
  snaps = collect(snap)
  nsnaps = get_nsnaps(snap)

  if size(snaps,1) < size(snaps,2)
    basis_space = tpod(snaps;kwargs...)
    compressed_time_snaps = change_mode(basis_space'*snaps,nsnaps)
    basis_time = tpod(compressed_time_snaps;kwargs...)
  else
    time_snaps = change_mode(snaps,nsnaps)
    basis_time = tpod(time_snaps;kwargs...)
    compressed_space_snaps = change_mode(basis_time'*time_snaps,nsnaps)
    basis_space = tpod(compressed_space_snaps;kwargs...)
  end

  basis_space,basis_time
end

for A in (:TimeAffinity,:ParamTimeAffinity)
  @eval begin
    function tpod(snap::Snapshots{$A};kwargs...)
      snaps = collect(snap)
      nsnaps = get_nsnaps(snap)
      time_ndofs = Int(size(snaps,2)/nsnaps)
      T = eltype(snaps)

      basis_space = tpod(snaps;kwargs...)
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

  nsols = nsnaps
  printstyled("Generating $nsols solution snapshots\n";color=:blue)

  collector = CollectSolutionsMap(fesolver,feop)
  sols = lazy_map(collector,params)
  Snapshots(collector,sols,nsols)
end

function collect_residuals(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  params::Table,
  args...;
  nsnaps=50)

  collector = CollectResidualsMap(fesolver,feop,args...)

  get_nress(::CollectResidualsMap) = nsnaps
  get_nress(::CollectResidualsMap{Union{TimeAffinity,NonAffinity}}) = 1
  nress = get_nress(collector)

  printstyled("Generating $nress residuals snapshots\n";color=:blue)
  sols = view(get_snaps(snaps),1:nress)
  params = view(params,1:nress)
  ress = lazy_map(collector,sols,params)

  Snapshots(collector,ress,nress)
end

function collect_jacobians(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  params::Table,
  args...;
  i=1,nsnaps=50)

  collector = CollectJacobiansMap(fesolver,feop,args...;i)

  get_njacs(::CollectJacobiansMap) = nsnaps
  get_njacs(::CollectJacobiansMap{Union{TimeAffinity,NonAffinity}}) = 1
  njacs = get_njacs(collector)

  printstyled("Generating $njacs jacobians snapshots\n";color=:blue)
  sols = view(get_snaps(snaps),1:njacs)
  params = view(params,1:njacs)
  jacs = lazy_map(collector,sols,params)

  Snapshots(collector,jacs,njacs)
end

function save(info::RBInfo,snap::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps")
    convert!(Matrix{Float},snap)
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
