struct Snapshots{A}
  snaps::LazyArray
  nsnaps::Int
  function Snapshots(collector::CollectorMap{A},snaps::LazyArray,nsnaps::Int) where A
    new{A}(collector,snaps,nsnaps)
  end
end

function lazy_collect(snap::Snapshots)
  snaps_dofs = get_snaps(snap)
  ApplyArray(hcat,snaps_dofs...)
end

Base.size(snap::Snapshots,idx...) = size(lazy_collect(snap))

get_snaps(snap::Snapshots) = snap.snaps

get_nsnaps(snap::Snapshots) = snap.nsnaps

get_time_ndofs(snap::Snapshots) = Int(size(snap.snaps,2)/snap.nsnaps)

function tpod(snap::Snapshots;kwargs...)
  snaps = lazy_collect(snap)
  nsnaps = get_nsnaps(snap)

  basis_space = tpod(snaps;kwargs...)
  compressed_time_snaps = change_mode(basis_space'*snaps,nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)

  basis_space,basis_time
end

for A in (:TimeAffinity,:ParamTimeAffinity)
  @eval begin
    function tpod(snap::Snapshots{$A};kwargs...)
      snaps = lazy_collect(snap)
      time_ndofs = get_time_ndofs(snap)
      T = eltype(snaps)

      basis_space = tpod(snaps;kwargs...)
      basis_time = ones(T,time_ndofs,1)
      basis_space,basis_time
    end
  end
end

function collect_solutions(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver)

  nsols = info.nsnaps_state
  params = realization(feop,nsols)
  printstyled("Generating $nsols solution snapshots\n";color=:blue)

  collector = CollectSolutionsMap(fesolver,feop)
  sols = lazy_map(collector,params)
  Snapshots(collector,sols,params)
end

function collect_residuals(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  trian::Triangulation,
  args...)

  sols = get_snaps(snaps)
  collector = CollectResidualsMap(fesolver,feop,trian)

  get_nress(::CollectResidualsMap) = info.nsnaps_system
  get_nress(::CollectResidualsMap{Union{TimeAffinity,NonAffinity}}) = 1
  nress = get_nress(collector)

  printstyled("Generating $nress residuals snapshots\n";color=:blue)
  sols = view(get_snaps(snaps),1:nress)
  ress = lazy_map(collector.f,sols,params)

  Snapshots(collector,ress,nress)
end

function collect_jacobians(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  trian::Triangulation,
  args...;
  i=1)

  sols = get_snaps(snaps)
  collector = CollectJacobiansMap(fesolver,feop,trian)

  get_njacs(::CollectJacobiansMap) = info.nsnaps_system
  get_njacs(::CollectJacobiansMap{Union{TimeAffinity,NonAffinity}}) = 1
  njacs = get_njacs(collector)

  printstyled("Generating $njacs jacobians snapshots\n";color=:blue)
  sols = view(get_snaps(snaps),1:njacs)
  jacs = lazy_map(collector.f,sols,params;i)

  Snapshots(collector,jacs,njacs)
end

function save(info::RBInfo,snaps::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps")
    convert!(Matrix{Float},snaps)
    save(path,snaps)
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
  s = load(T,path)
  s
end

function load(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params")
  load(T,path)
end
