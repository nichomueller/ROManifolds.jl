struct Snapshots
  snaps::AbstractMatrix{Float}
  params::Table{Float,Param,Vector{Int32}}
  nonzero_idx::Vector{Int32}
end

get_snaps(s::Snapshots) = s.snaps

get_params(s::Snapshots) = s.params

Base.length(s::Snapshots) = length(get_params(s))

function generate_fe_snapshots(feop,solver,nsnap)
  sols = solve(solver,feop,nsnap)
  cache,nonzero_idx = snapshot_cache(solver,feop,nsnap)
  snaps = pmap(sol->get_solution!(cache,sol),sols)
  mat_snaps = EMatrix(first.(snaps))
  param_snaps = Table(last.(snaps))
  Snapshots(mat_snaps,param_snaps,nonzero_idx)
end

function get_solution!(cache,sol::ParamFESolution)
  sol_cache,param_cache = cache

  k = sol.psol.k
  printstyled("Computing snapshot $k\n";color=:blue)
  copyto!(sol_cache,get_free_dof_values(sol.psol.uh))
  copyto!(param_cache,get_param(sol.psol.μ))
  printstyled("Successfully computed snapshot $k\n";color=:blue)

  sol_cache,param_cache
end

function get_solution!(cache,sol::ParamTransientFESolution)
  sol_cache,param_cache = cache

  k = sol.psol.k
  printstyled("Computing snapshot $k\n";color=:blue)
  n = 1
  for (xn,_) in sol
    setindex!(sol_cache,xn,:,n)
    n += 1
  end
  copyto!(param_cache,get_param(sol.psol.μ))
  printstyled("Successfully computed snapshot $k\n";color=:blue)

  sol_cache,param_cache
end

# function get_dirichlet_values(
#   U::ParamTrialFESpace,
#   μ::Vector{Param})

#   nsnap = length(μ)
#   dir(μ) = U(μ).dirichlet_values
#   Snapshots(:g,dir.(μ),nsnap,EMatrix{Float})
# end

# function get_dirichlet_values(
#   U::ParamTransientTrialFESpace,
#   μ::Vector{Param},
#   tinfo::TimeInfo)

#   nsnap = length(μ)
#   times = get_times(tinfo)
#   dir(μ) = Matrix([U(μ,t).dirichlet_values for t=times])
#   Snapshots(:g,dir.(μ),nsnap,EMatrix{Float})
# end

# function online_loop(fe_sol,rb_space,rb_system,k::Int)
#   online_time = @elapsed begin
#     lhs,rhs = rb_system(k)
#     rb_sol = solve_rb_system(lhs,rhs)
#   end
#   fe_sol_approx = reconstruct_fe_sol(rb_space,rb_sol)

#   RBResults(fe_sol,fe_sol_approx,online_time)
# end

# function online_loop(loop,k::UnitRange{Int})
#   pmap(loop,k)
# end
