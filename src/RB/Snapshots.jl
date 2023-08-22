struct Snapshots{C,T,N}
  collector::C
  snaps::LazyArray{T,N}
  params::Table
  function Snapshots(collector::C,snaps::LazyArray{T,N},params::Table) where {C,T,N}
    new{C,T,N}(collector,snaps,params)
  end
end

get_snaps(snap::Snapshots) = snap.snaps

get_params(snap::Snapshots,idx=:) = snap.params[idx]

get_nsnaps(snap::Snapshots) = length(snap.params)

function lazy_collect(snap::Snapshots)
  snaps_dofs = get_snaps(snap)
  ApplyArray(hcat,snaps_dofs...)
end

function lazy_collect(snap::Snapshots{CollectSolutionsMap,T,N} where {T,N})
  snaps = get_snaps(snap)
  f = Broadcasting(get_free_dof_values)
  snaps_dofs = lazy_map(f,snaps)
  ApplyArray(hcat,snaps_dofs...)
end

function tpod(snap::Snapshots;kwargs...)
  snaps = lazy_collect(snap)
  nsnaps = get_nsnaps(snap)

  basis_space = tpod(snaps;kwargs...)
  compressed_time_snaps = change_mode(basis_space'*snaps,nsnaps)
  basis_time = tpod(compressed_time_snaps;kwargs...)

  basis_space,basis_time
end

# function tpod(snap::Snapshots;kwargs...)
#   snaps = lazy_collect(snap)
#   mode = Val(size(snaps,1) < size(snaps,2))
#   tpod(mode,snap;kwargs...)
# end

# function tpod(::Val{true},snap::Matrix{T};kwargs...) where T
#   snaps,nsnaps = get_snaps(snap),get_nsnaps(snap)

#   basis_space = tpod(snaps;kwargs...)
#   compressed_time_snaps = change_mode(basis_space'*snaps,nsnaps)
#   basis_time = tpod(compressed_time_snaps;kwargs...)

#   basis_space,basis_time
# end

# function tpod(::Val{false},snap::Matrix{T};kwargs...) where T
#   snaps,nsnaps = get_snaps(snap),get_nsnaps(snap)

#   time_snaps = change_mode(snaps,nsnaps)
#   basis_time = tpod(time_snaps)
#   compressed_space_snaps = change_mode(basis_time'*time_snaps,nsnaps)
#   basis_space = tpod(compressed_space_snaps;kwargs...)

#   basis_space,basis_time
# end

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

  nres = info.nsnaps_system
  sols = get_snaps(snaps)
  cache = array_cache(sols)

  function run_collection(collector::CollectResidualsMap)
    params = get_params(sols,1:nres)
    printstyled("Generating $nres residuals snapshots\n";color=:blue)
    ress = lazy_map(eachindex(params)) do i
      sol_i = getindex!(cache,sols,i)
      param_i = getindex(params,i)
      collector.f(sol_i,param_i)
    end
    return ress,params
  end

  function run_collection(collector::CollectResidualsMap{Union{TimeAffinity,NonAffinity}})
    params = get_params(sols,1)
    printstyled("Generating 1 residual snapshot\n";color=:blue)
    ress = lazy_map(eachindex(params)) do i
      sol_i = getindex!(cache,sols,i)
      param_i = getindex(params,i)
      collector.f(sol_i,param_i)
    end
    return ress,params
  end

  collector = CollectResidualsMap(fesolver,feop,trian)
  ress,params = lazy_map(collector,sols)
  Snapshots(collector,ress,params)
end

function collect_jacobians(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  snaps::Snapshots,
  trian::Triangulation,
  args...;
  i=1)

  njac = info.nsnaps_system
  sols = get_snaps(snaps)
  cache = array_cache(sols)

  function run_collection(collector::CollectJacobiansMap)
    params = get_params(sols,1:njac)
    printstyled("Generating $njac jacobians snapshots\n";color=:blue)
    jacs = lazy_map(eachindex(params)) do i
      sol_i = getindex!(cache,sols,i)
      param_i = getindex(params,i)
      collector.f(sol_i,param_i)
    end
    return jacs,params
  end

  function run_collection(collector::CollectJacobiansMap{Union{TimeAffinity,NonAffinity}})
    params = get_params(sols,1)
    printstyled("Generating 1 jacobian snapshot\n";color=:blue)
    jacs = lazy_map(eachindex(params)) do i
      sol_i = getindex!(cache,sols,i)
      param_i = getindex(params,i)
      collector.f(sol_i,param_i)
    end
    return jacs,params
  end

  collector = CollectJacobiansMap(fesolver,feop,trian;i)
  jacs,params = run_collection(collector)
  Snapshots(collector,jacs,params)
end

function Arrays.lazy_map(
  k::CollectResidualsMap,
  sols::Snapshots)

  cache = array_cache(sols)

  lazy_map(k,first(sols))
end

function Arrays.lazy_map(
  k::CollectResidualsMap{Union{TimeAffinity,NonAffinity}},
  sols::Snapshots)

  lazy_map(k,first(sols))
end

function Arrays.lazy_map(
  k::CollectJacobiansMap{Union{TimeAffinity,NonAffinity}},
  sols::Snapshots)

  lazy_map(k,first(sols))
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
