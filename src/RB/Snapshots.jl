struct Snapshots{T,A}
  snaps::LazyArray
  nsnaps::Int
  function Snapshots(::CollectorMap{A},snaps::LazyArray,nsnaps::Int) where A
    T = eltype(snaps)
    new{T,A}(snaps,nsnaps)
  end
end

const SingleFieldSnapshots{A} = Snapshots{<:AbstractArray,A}
const MultiFieldSnapshots{A} = Snapshots{<:BlockArray,A}

function Base.collect(snap::Snapshots)
  lazy_snaps = get_snaps(snap)
  collect(lazy_snaps)
end

# DO NOT RECOMMEND USING
function Base.size(snap::Snapshots)
  len = snap.nsnaps
  s1 = first(snap.snaps)
  size(s1,1),size(s1,2)*len
end

get_snaps(snap::Snapshots) = snap.snaps

get_nsnaps(snap::Snapshots) = snap.nsnaps

get_time_ndofs(snap::Snapshots) = size(first(snap.snaps),2)

function tpod(snap::Snapshots;kwargs...)
  s = size(snap)
  tpod(Val(s[1] < s[2]),snap;kwargs...)
end

function tpod(::Val{true},snap::Snapshots;kwargs...)
  snaps = collect(snap)
  nsnaps = get_nsnaps(snap)

  basis_space = tpod(snaps;kwargs...)
  compressed_space_snaps = prod(basis_space,snaps)
  compressed_time_snaps = change_mode(compressed_space_snaps,nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)
  basis_space,basis_time
end

function tpod(::Val{false},snap::Snapshots;kwargs...)
  snaps_t = collect(lazy_map(transpose,get_snaps(snap)))
  nsnaps = get_nsnaps(snap)

  basis_time = tpod(snaps_t;kwargs...)
  compressed_time_snaps = prod(basis_time,snaps_t)
  compressed_space_snaps = change_mode(compressed_time_snaps,nsnaps)
  basis_space = tpod(compressed_space_snaps;kwargs...)
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
