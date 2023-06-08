function generate_snapshots(feop,solver,nsnap)
  sols = solve(solver,feop,nsnap)
  cache = snapshots_cache(feop,solver)
  snaps = pmap(sol->collect_snapshot!(cache,sol),sols)
  hcat!(first.(snaps))
  param_snaps = Table(last.(snaps))
  Snapshots(mat_snaps,param_snaps)
end

abstract type Snapshots{T} end

mutable struct SingleFieldSnapshots{T} <: Snapshots{T}
  snaps::NnzMatrix{T}
  params::Table
end

mutable struct MultiFieldSnapshots{T} <: Snapshots{T}
  snaps::Vector{NnzMatrix{T}}
  params::Table
end

Base.size(s::Snapshots) = size(s.snaps)

Base.size(s::MultiFieldSnapshots) = size.(s.snaps)

Base.length(s::Snapshots) = length(s.params)

nfields(s::MultiFieldSnapshots) = length(s.snaps)

get_snaps(s::Snapshots) = s.snaps

get_params(s::Snapshots) = s.params

is_transient(s::Snapshots) = all(tndofs -> tndofs > 1,_complementary_dimension(s))

function Snapshots(snaps::AbstractMatrix{Float},params::Table)
  SingleFieldSnapshots(snaps,params)
end

function Snapshots(snaps::Vector{<:AbstractMatrix{Float}},params::Table)
  MultiFieldSnapshots(snaps,params)
end

function get_single_field(s::MultiFieldSnapshots,fieldid::Int)
  Snapshots(s.snaps[fieldid],s.params)
end

function collect_single_fields(s::MultiFieldSnapshots)
  map(fieldid -> get_single_field(s,fieldid),1:nfields(s))
end

function snapshots_cache(feop,args...)
  param_cache = realization(feop)
  sol_cache = solution_cache(feop.test,args...)
  sol_cache,param_cache
end

function solution_cache(test::FESpace,::FESolver)
  space_ndofs = test.nfree
  cache = fill(1.,space_ndofs,1)
  NnzMatrix(cache)
end

function solution_cache(test::FESpace,solver::ODESolver)
  space_ndofs = test.nfree
  time_ndofs = get_time_ndofs(solver)
  cache = fill(1.,space_ndofs,time_ndofs)
  NnzMatrix(cache)
end

function solution_cache(test::MultiFieldFESpace,solver::GridapType)
  map(t->solution_cache(t,solver),test.spaces)
end

function collect_snapshot!(cache,sol::ParamSolution)
  sol_cache,param_cache = cache

  printstyled("Computing snapshot $(sol.k)\n";color=:blue)
  copyto!(sol_cache,sol.uh)
  copyto!(param_cache,sol.μ)
  printstyled("Successfully computed snapshot $(sol.k)\n";color=:blue)

  sol_cache,param_cache
end

function collect_snapshot!(cache,sol::ParamODESolution)
  sol_cache,param_cache = cache

  printstyled("Computing snapshot $(sol.k)\n";color=:blue)
  n = 1
  for (xn,_) in sol
    setindex!(sol_cache,xn,:,n)
    n += 1
  end
  copyto!(param_cache,sol.μ)
  printstyled("Successfully computed snapshot $(sol.k)\n";color=:blue)

  sol_cache,param_cache
end

_complementary_dimension(s::SingleFieldSnapshots) = Int(size(s,2)/length(s))

function _complementary_dimension(s::MultiFieldSnapshots)
  single_fields = collect_single_fields(s)
  _complementary_dimension.(single_fields)
end

function change_mode!(s::SingleFieldSnapshots)
  mode1_ndofs = size(s,1)
  mode2_ndofs = _complementary_dimension(s)
  nparams = length(s)
  change_mode!(s.snaps,mode1_ndofs,mode2_ndofs,nparams)
end

function change_mode!(s::MultiFieldSnapshots)
  mode1_ndofs = size(s,1)
  mode2_ndofs = _complementary_dimension(s)
  nparams = length(s)
  map((snap,m1ndfs,m2ndfs) -> change_mode!(snap,m1ndfs,m2ndfs,nparams),
    s.snaps,mode1_ndofs,mode2_ndofs)
end

function tpod(s::SingleFieldSnapshots;kwargs...)
  tpod(s.snaps;kwargs...)
end

function transient_tpod(::Val{false},s::SingleFieldSnapshots;kwargs...)
  s1 = copy(s)
  basis_space = tpod(s1;kwargs...)
  s2 = basis_space'*s
  change_mode!(s2)
  basis_time = tpod(s2;kwargs...)

  basis_space,basis_time
end

function transient_tpod(s::SingleFieldSnapshots;kwargs...)
  compress_rows = _compress_rows(s.snaps)
  transient_tpod(compress_rows,s;kwargs...)
end

function transient_tpod(::Val{true},s::SingleFieldSnapshots;kwargs...)
  s1 = copy(s)
  change_mode!(s1)
  basis_time = tpod(s1;kwargs...)
  s2 = basis_time'*s
  change_mode!(s2)
  basis_space = tpod(s2;kwargs...)

  basis_space,basis_time
end

function save(info::RBInfo,ref::Symbol,snaps::Snapshots)
  if info.save_offline
    path = joinpath(info.fe_path,ref)
    save(path,snaps)
  end
end

function load(T::Type{Snapshots},info::RBInfo,ref::Symbol)
  path = joinpath(info.fe_path,ref)
  load(T,path)
end
