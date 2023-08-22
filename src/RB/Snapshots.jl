struct Snapshots{A,T,N}
  snaps::LazyArray{T,N}
  params::Table
  function Snapshots(::A,snaps::LazyArray{T,N},params::Table) where {A,T,N}
    new{A,T,N}(snaps,params)
  end
end

get_snaps(snap::Snapshots,idx=:) = snap.snaps[idx]

get_params(snap::Snapshots,idx=:) = snap.params[idx]

get_nsnaps(snap::Snapshots) = length(snap.params)

get_time_ndofs(snap::Snapshots) = Int(size(get_snaps(snap),2)/get_nsnaps(snap))

function tpod(snap::Snapshots;kwargs...)
  mode = Val(size(s.snaps,1) > size(s.snaps,2))
  tpod(mode,snap;kwargs...)
end

function tpod(::Val{false},snap::Snapshots;kwargs...)
  snaps,nsnaps = get_snaps(snap),get_nsnaps(snap)

  basis_space = tpod(snaps;kwargs...)
  compressed_time_snaps = change_mode(basis_space'*snaps,nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)

  basis_space,basis_time
end

function tpod(::Val{true},snap::Snapshots;kwargs...)
  snaps,nsnaps = get_snaps(snap),get_nsnaps(snap)

  time_snaps = change_mode(snaps,nsnaps)
  basis_time = tpod(time_snaps)
  compressed_space_snaps = change_mode(basis_time'*time_snaps,nsnaps)
  basis_space = tpod(compressed_space_snaps;kwargs...)

  basis_space,basis_time
end

for A in (:TimeAffinity,:ParamTimeAffinity)
  @eval begin
    function tpod(snaps::Snapshots{T,N,$A} where N;kwargs...) where T
      snaps,time_ndofs = get_snaps(snap),get_time_ndofs(snap)

      basis_space = tpod(snaps;kwargs...)
      basis_time = ones(T,time_ndofs,1)
      basis_space,basis_time
    end
  end
end

struct ParamPair{T,N}
  snap::AbstractArray{T,N}
  param::Table
end

get_snap(pp::ParamPair) = pp.snap

get_param(pp::ParamPair) = pp.param

function Base.getindex(snap::Snapshots{T,N,A} where A,idx::Int) where {T,N}
  return ParamPair{T,N}(get_snaps(snap,idx),get_params(snap,idx))
end

function Base.iterate(snap::Snapshots)
  state = 1
  return snap[state],state+1
end

function Base.iterate(snap::Snapshots,state::Int)
  if state > get_nsnaps(snap)
    return nothing
  end
  return snap[state],state+1
end

function collect_solutions(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver)

  nsols = info.nsnaps_state
  params = realization(feop,nsols)
  printstyled("Generating $nsols solution snapshots\n";color=:blue)

  collect = CollectSolutionsMap(fesolver,feop)
  sols = lazy_map(collect,params)
  Snapshots(NonAffinity(),sols,params)
end

function collect_residuals(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  sols::Snapshots,
  args...)

  nres = info.nsnaps_system
  params = get_params(sols)
  aff = affinity_residual(feop,fesolver,sols,args...)
  printstyled("Generating $nres residual snapshots, affinity: $aff\n";color=:blue)

  collect = CollectResidualsMap(fesolver,feop)
  ress = lazy_map(collect,sols)
  Snapshots(aff,ress,params)
end

function collect_jacobians(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  sols::Snapshots,
  args...)

  njac = info.nsnaps_system
  params = get_params(sols)
  aff = affinity_jacobian(feop,fesolver,sols,args...)
  printstyled("Generating $njac jacobians snapshots, affinity: $aff\n";color=:blue)

  collect = CollectJacobiansMap(fesolver,feop)
  jacs = lazy_map(collect,sols)
  Snapshots(aff,jacs,params)
end
