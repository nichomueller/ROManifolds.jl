struct RBResults{T}
  name::Symbol
  params::Table
  sol::PTArray{Matrix{T}}
  sol_approx::PTArray{Matrix{T}}
  relative_err::Vector{Float}
  wall_time::Float
  nallocations::Float

  function RBResults(
    params::Table,
    sol::PTArray{Matrix{T}},
    sol_approx::PTArray{Matrix{T}},
    stats::NamedTuple;
    name=:vel,
    kwargs...) where T

    relative_err = compute_relative_error(sol,sol_approx;kwargs...)
    wall_time = stats[:time]
    nallocations = stats[:bytes]/1e6
    new{T}(name,params,sol,sol_approx,relative_err,wall_time,nallocations)
  end
end

Base.length(r::RBResults) = length(r.params)
get_avg_error(r::RBResults) = sum(r.relative_err) / length(r)
get_avg_time(r::RBResults) = r.wall_time / length(r)
get_avg_nallocs(r::RBResults) = r.nallocations / length(r)

function Base.show(io::IO,r::RBResults)
  name = r.name
  avg_err = get_avg_error(r)
  avg_time = get_avg_time(r)
  avg_nallocs = get_avg_nallocs(r)
  print(io,"Average online relative errors for $name: $avg_err\n")
  print(io,"Average online wall time: $avg_time [s]\n")
  print(io,"Average number of allocations: $avg_nallocs [Mb]\n")
end

function Base.first(r::RBResults)
  name = r.name
  μ = r.params[1]
  sol = r.sol[1]
  sol_approx = r.sol_approx[1]
  relative_err = get_avg_error(r)
  wall_time = get_avg_time(r)
  nallocations = get_avg_nallocs(r)
  name,μ,sol,sol_approx,relative_err,wall_time,nallocations
end

function save(info::RBInfo,r::RBResults)
  if info.save_structures
    path = joinpath(info.rb_path,"results")
    save(path,r)
  end
end

function load(info::RBInfo,T::Type{RBResults})
  path = joinpath(info.rb_path,"results")
  load(path,T)
end

function post_process(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  sol::PTArray,
  params::Table,
  sol_approx::PTArray,
  stats::NamedTuple)

  nparams = length(params)
  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(feop,energy_norm)
  _sol = space_time_matrices(sol;nparams)
  _sol_approx = space_time_matrices(sol_approx;nparams)
  results = RBResults(params,_sol,_sol_approx,stats;norm_matrix)
  show(results)
  save(info,results)
  writevtk(info,feop,fesolver,results)
  return
end

function allocate_online_cache(
  feop::PTFEOperator,
  fesolver::PODESolver,
  snaps_test::PTArray{Vector{T}},
  params::Table) where T

  times = get_times(fesolver)
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,params,times)
  b = allocate_residual(ode_op,params,times,snaps_test,ode_cache)
  A = allocate_jacobian(ode_op,params,times,snaps_test,ode_cache)

  coeff = zeros(T,1,1)
  ptcoeff = PTArray([zeros(T,1,1) for _ = eachindex(params)])

  res_contrib_cache = return_cache(RBVecContributionMap(T))
  jac_contrib_cache = return_cache(RBMatContributionMap(T))

  res_cache = (b,CachedArray(coeff),CachedArray(ptcoeff)),res_contrib_cache
  jac_cache = (A,CachedArray(coeff),CachedArray(ptcoeff)),jac_contrib_cache
  res_cache,jac_cache
end

function test_rb_solver(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PODESolver,
  rbspace,
  rbres,
  rbjacs,
  snaps,
  params::Table)

  snaps_test,params_test = load_test(info,feop,fesolver)

  println("Solving linear RB problems")
  x = initial_guess(snaps,params,params_test)
  x .= recenter(fesolver,x,μ)
  rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
  rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbres,rbspace,x,params_test)
  lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rbjacs,rbspace,x,params_test)

  stats = @timed begin
    rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs)
  end
  approx_snaps_test = recast(rb_snaps_test,rbspace)
  post_process(info,feop,fesolver,snaps_test,params_test,approx_snaps_test,stats)
end

function test_rb_solver(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace,
  rbres,
  rbjacs,
  snaps,
  params::Table)

  snaps_test,params_test = load_test(info,feop,fesolver)

  println("Solving nonlinear RB problems with Newton iterations")
  rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
  nl_cache = nothing
  x = initial_guess(snaps,params,params_test)
  xrb = space_time_projection(x,rbspace)
  _,conv0 = Algebra._check_convergence(fesolver.nls.ls,xrb)
  stats = @timed begin
    for iter in 1:fesolver.nls.max_nliters
      x .= recenter(fesolver,x,params_test)
      rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbres,rbspace,x,params_test)
      lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rbjacs,rbspace,x,params_test)
      nl_cache = rb_solve!(xrb,fesolver.nls.ls,rhs,lhs,nl_cache)
      x .= recast(xrb,rbspace)
      isconv,conv = Algebra._check_convergence(fesolver.nls,xrb,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
      if all(isconv); return; end
      if iter == nls.max_nliters
        @unreachable
      end
    end
  end
  post_process(info,feop,fesolver,snaps_test,params_test,x,stats)
end

function load_test(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  ntests = info.nsnaps_test
  try
    @check info.load_solutions
    sols,params = try
      load_test(info,(Snapshots,Table))
    catch
      load_test(info,(BlockSnapshots,Table))
    end
    return sols[1:ntests],params[1:ntests]
  catch
    params = realization(feop,ntests)
    trial = get_trial(feop)
    sols, = collect_solutions(fesolver,feop,trial,params)
    save_test(info,(sols,params))
    return sols[1:ntests],params[1:ntests]
  end
end

function rb_solve(ls::LinearSolver,rhs::PTArray,lhs::PTArray)
  x = copy(rhs)
  cache = nothing
  rb_solve!(x,ls,rhs,lhs,cache)
  return x
end

function rb_solve!(
  x::PTArray,
  ls::LinearSolver,
  rhs::PTArray,
  lhs::PTArray,
  ::Nothing)

  ss = symbolic_setup(ls,testitem(lhs))
  ns = numerical_setup(ss,lhs)
  _rb_loop_solve!(x,ns,rhs)
  return ns
end

function rb_solve!(
  x::PTArray,
  ::LinearSolver,
  rhs::PTArray,
  lhs::PTArray,
  ns)

  numerical_setup!(ns,lhs)
  _rb_loop_solve!(x,ns,rhs)
  return ns
end

function _rb_loop_solve!(x::PTArray,ns,b::PTArray)
  @inbounds for k in eachindex(x)
    solve!(x[k],ns[k],-b[k])
  end
  x
end

function initial_guess(
  sols::Snapshots,
  params::Table,
  params_test::Table)

  kdtree = KDTree(map(x -> SVector(Tuple(x)),params))
  idx_dist = map(x -> nn(kdtree,SVector(Tuple(x))),params_test)
  sols[first.(idx_dist)]
end

function space_time_matrices(sol::PTArray{Vector{T}};nparams=length(sol)) where T
  mat = hcat(get_array(sol)...)
  ntimes = Int(size(mat,2)/nparams)
  array = Vector{Matrix{eltype(T)}}(undef,nparams)
  @inbounds for i = 1:nparams
    array[i] = mat[:,(i-1)*ntimes+1:i*ntimes]
  end
  PTArray(array)
end

function compute_relative_error(
  sol::PTArray{Matrix{T}},
  sol_approx::PTArray{Matrix{T}};
  kwargs...) where T

  @assert length(sol) == length(sol_approx)

  time_ndofs = length(testitem(sol))
  nparams = length(sol)
  ncache = zeros(T,time_ndofs)
  dcache = zeros(T,time_ndofs)
  cache = ncache,dcache
  err = Vector{T}(undef,nparams)
  @inbounds for i = 1:nparams
    erri = compute_relative_error!(cache,sol[i],sol_approx[i];kwargs...)
    err[i] = erri
  end
  err
end

function compute_relative_error!(cache,sol,sol_approx;norm_matrix=nothing)
  ncache,dcache = cache
  @inbounds for i = axes(sol,2)
    ncache[i] = norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    dcache[i] = norm(sol[:,i],norm_matrix)
  end
  norm(ncache)/norm(dcache)
end

LinearAlgebra.norm(v::AbstractVector,::Nothing) = norm(v)

LinearAlgebra.norm(v::AbstractVector,X::AbstractMatrix) = v'*X*v

function Gridap.Visualization.writevtk(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  results::RBResults)

  test = get_test(feop)
  trial = get_trial(feop)
  trian = get_triangulation(test)
  times = get_times(fesolver)

  name,μ,sol,sol_approx, = first(results)
  pointwise_err = abs.(sol-sol_approx)

  plt_dir = joinpath(info.rb_path,"plots")
  create_dir!(plt_dir)
  for (it,t) in enumerate(times)
    fsol = FEFunction(trial(μ,t),sol[:,it])
    fsol_approx = FEFunction(trial(μ,t),sol_approx[:,it])
    ferr = FEFunction(trial(μ,t),pointwise_err[:,it])
    writevtk(trian,joinpath(plt_dir,"$(name)_$(it).vtu"),cellfields=["$name"=>fsol])
    writevtk(trian,joinpath(plt_dir,"$(name)_approx_$(it).vtu"),cellfields=["$(name)_approx"=>fsol_approx])
    writevtk(trian,joinpath(plt_dir,"$(name)_err_$(it).vtu"),cellfields=["$(name)_err"=>ferr])
  end
end

function save_test(info::RBInfo,snaps::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps_test")
    save(path,snaps)
  end
end

function save_test(info::RBInfo,params::Table)
  if info.save_structures
    path = joinpath(info.fe_path,"params_test")
    save(path,params)
  end
end

function load_test(info::RBInfo,T::Type{Snapshots})
  path = joinpath(info.fe_path,"fesnaps_test")
  load(path,T)
end

function load_test(info::RBInfo,T::Type{Table})
  path = joinpath(info.fe_path,"params_test")
  load(path,T)
end

function save(info::RBInfo,result::RBResults)
  path = joinpath(info.rb_path,"rbresults")
  save(path,result)
end

function load(info::RBInfo,T::Type{RBResults})
  path = joinpath(info.rb_path,"rbresults")
  load(path,T)
end
