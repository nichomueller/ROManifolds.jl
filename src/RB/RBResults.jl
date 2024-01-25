function TransientFETools.allocate_cache(op::NonlinearOperator,rbspace)
  T = eltype(get_vector_type(rbspace))
  N = length(op.μ)
  b = allocate_residual(op,op.u0)
  A = allocate_jacobian(op,op.u0)
  mat = zeros(T,1,1)
  cmat = CachedArray(mat)
  coeff = CachedArray(mat)
  ptcoeff = CachedArray(zero_param_array(mat,N))

  res_contrib_cache = return_cache(RBVecContributionMap(),op.u0)
  jac_contrib_cache = return_cache(RBMatContributionMap(),op.u0)

  rb_ndofs = num_rb_ndofs(rbspace)

  rhs_solve_cache = zero_param_array(zeros(T,rb_ndofs),N)
  lhs_solve_cache = zero_param_array(zeros(T,rb_ndofs,rb_ndofs),N)

  res_cache = ((b,cmat),(coeff,ptcoeff)),res_contrib_cache
  jac_cache = ((A,cmat),(coeff,ptcoeff)),jac_contrib_cache
  (res_cache,jac_cache),(rhs_solve_cache,lhs_solve_cache)
end

function rb_solver(rbinfo,feop::TransientParamFEOperator{Affine},fesolver,rbspace,rbres,rbjacs,snaps,params)
  println("Solving linear RB problems")
  nsnaps_test = rbinfo.nsnaps_test
  snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
  op = get_method_operator(fesolver,feop,snaps_test,params_test)
  cache,(_,lhs) = allocate_cache(op,rbspace)
  stats = @timed begin
    rhs,(_lhs,_lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op,rbres,rbjacs,rbspace)
    @. lhs = _lhs+_lhs_t
    rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs)
  end
  approx_snaps_test = recast(rb_snaps_test,rbspace)
  post_process(rbinfo,feop,snaps_test,approx_snaps_test,params_test,stats)
end

function rb_solver(rbinfo,feop_lin,feop_nlin,fesolver,rbspace,rbres,rbjacs,snaps,params;tol=rbinfo.ϵ)
  println("Solving nonlinear RB problems with Newton iterations")
  nsnaps_test = rbinfo.nsnaps_test
  snaps_train,params_train = snaps[1:nsnaps_test],params[1:nsnaps_test]
  snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
  x = nearest_neighbor(snaps_train,params_train,params_test)
  op_lin = get_method_operator(fesolver,feop_lin,snaps_test,params_test)
  op_nlin = get_method_operator(fesolver,feop_nlin,snaps_test,params_test)
  xrb = space_time_projection(x,rbspace)
  dxrb = similar(xrb)
  cache,(rhs,lhs) = allocate_cache(op,rbspace)
  newt_cache = nothing
  uh0_test = fesolver.uh0(params_test)
  conv0 = ones(nsnaps_test)
  rbrhs_lin,rbrhs_nlin = rbres
  rblhs_lin,rblhs_nlin = rbjacs
  stats = @timed begin
    rhs_lin,(lhs_lin,lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op_lin,rbrhs_lin,rblhs_lin,rbspace)
    for iter in 1:fesolver.nls.max_nliters
      x = recenter(x,uh0_test;θ=fesolver.θ)
      # op_nlin = update_method_operator(op_nlin,x)
      rhs_nlin,(lhs_nlin,) = collect_rhs_lhs_contributions!(cache,rbinfo,op_nlin,rbrhs_nlin,rblhs_nlin,rbspace)
      @. lhs = lhs_lin+lhs_t+lhs_nlin
      @. rhs = rhs_lin+rhs_nlin+(lhs_lin+lhs_t)*xrb
      newt_cache = rb_solve!(dxrb,fesolver.nls.ls,rhs,lhs,newt_cache)
      xrb += dxrb
      x = recast(xrb,rbspace)
      isconv,conv = Algebra._check_convergence(tol,dxrb,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
      if all(isconv); break; end
      if iter == fesolver.nls.max_nliters
        @unreachable
      end
    end
  end
  post_process(rbinfo,feop,snaps_test,x,params_test,stats)
end

function rb_solve(ls::LinearSolver,rhs::ParamArray,lhs::ParamArray)
  x = copy(rhs)
  cache = nothing
  rb_solve!(x,ls,rhs,lhs,cache)
  return x
end

function rb_solve!(
  x::ParamArray,
  ls::LinearSolver,
  rhs::ParamArray,
  lhs::ParamArray,
  ::Nothing)

  ss = symbolic_setup(ls,testitem(lhs))
  ns = numerical_setup(ss,lhs)
  _rb_loop_solve!(x,ns,rhs)
  return ns
end

function rb_solve!(
  x::ParamArray,
  ::LinearSolver,
  rhs::ParamArray,
  lhs::ParamArray,
  ns)

  numerical_setup!(ns,lhs)
  _rb_loop_solve!(x,ns,rhs)
  return ns
end

function _rb_loop_solve!(x::ParamArray,ns,b::ParamArray)
  @inbounds for k in eachindex(x)
    solve!(x[k],ns[k],-b[k])
  end
  x
end

function nearest_neighbor(
  sols_train::ParamArray{T},
  params_train::Table,
  params_test::Table) where T

  nparams_train = length(params_train)
  ntimes = Int(length(sols_train)/nparams_train)
  kdtree = KDTree(map(x -> SVector(Tuple(x)),params_train))
  idx, = map(x -> nn(kdtree,SVector(Tuple(x))),params_test) |> tuple_of_arrays
  nparams_test = length(params_test)
  array = Vector{T}(undef,nparams_test*ntimes)
  @inbounds for n = 1:nparams_test
    idxn = idx[n]
    array[(n-1)*ntimes+1:n*ntimes] = sols_train[(idxn-1)*ntimes+1:idxn*ntimes]
  end
  ParamArray(array)
end

struct RBResults{T}
  name::Symbol
  params::Table
  sol::AbstractVector
  sol_approx::AbstractVector
  relative_err::Vector{T}
  fem_stats::ComputationInfo
  rb_stats::ComputationInfo
end

function RBResults(
  params::Table,
  sol::ParamArray,
  sol_approx::ParamArray,
  fem_stats::ComputationInfo,
  rb_stats::ComputationInfo,
  args...;
  name=:vel)

  nparams = length(params)
  ntimes = Int(length(sol)/nparams)
  _sol,_sol_approx = map(eachindex(params)) do np
    idx = (np-1)*ntimes+1:np*ntimes
    sol[idx],sol_approx[idx]
  end |> tuple_of_arrays
  relative_err = compute_relative_error(_sol,_sol_approx,args...)
  RBResults(name,params,_sol,_sol_approx,relative_err,fem_stats,rb_stats)
end

Base.length(r::RBResults) = length(r.params)
get_name(r::RBResults) = r.name
get_avg_error(r::RBResults) = sum(r.relative_err) / length(r)
get_speedup_time(r::RBResults) = get_avg_time(r.fem_stats) / get_avg_time(r.rb_stats)
get_speedup_memory(r::RBResults) = get_avg_nallocs(r.fem_stats) / get_avg_nallocs(r.rb_stats)

function Base.show(io::IO,r::RBResults)
  name = get_name(r)
  avg_err = get_avg_error(r)
  print(io,"Average online relative errors for $name: $avg_err\n")
  show_speedup(io,r)
end

function show_speedup(io::IO,r::RBResults)
  avg_time = get_avg_time(r.rb_stats)
  avg_nallocs = get_avg_nallocs(r.rb_stats)
  speedup_time = Float16(get_speedup_time(r))
  speedup_memory = Float16(get_speedup_memory(r))
  print(io,"Average online wall time: $avg_time [s]\n")
  print(io,"Average number of allocations: $avg_nallocs [Mb]\n")
  print(io,"FEM/RB wall time speedup: $speedup_time\n")
  print(io,"FEM/RB memory speedup: $speedup_memory\n")
end

function Utils.save(rbinfo::RBInfo,r::RBResults)
  name = get_name(r)
  path = joinpath(rbinfo.rb_path,"results_$name")
  save(path,r)
end

function Utils.load(rbinfo::RBInfo,T::Type{RBResults};name=:vel)
  path = joinpath(rbinfo.rb_path,"results_$name")
  load(path,T)
end

function post_process(
  rbinfo::RBInfo,
  feop::TransientParamFEOperator,
  sol::ParamArray,
  sol_approx::ParamArray,
  params::Table,
  stats::NamedTuple;
  show_results=true)

  nparams = length(params)
  norm_matrix = get_norm_matrix(rbinfo,feop)
  fem_stats = load(rbinfo,ComputationInfo)
  rb_stats = ComputationInfo(stats,nparams)
  results = RBResults(params,sol,sol_approx,fem_stats,rb_stats,norm_matrix)
  if show_results
    show(results)
  end
  return results
end

function compute_relative_error(
  sol::AbstractVector{<:ParamArray},
  sol_approx::AbstractVector{<:ParamArray},
  args...)

  @assert length(sol) == length(sol_approx)

  time_ndofs = length(testitem(sol))
  nparams = length(sol)
  ncache = zeros(T,time_ndofs)
  dcache = zeros(T,time_ndofs)
  cache = ncache,dcache

  map(1:nparams) do i
    compute_relative_error!(cache,sol[i],sol_approx[i],args...)
  end
end

function compute_relative_error!(cache,sol,sol_approx,norm_matrix=nothing)
  ncache,dcache = cache
  @inbounds for i = axes(sol,2)
    ncache[i] = norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    dcache[i] = norm(sol[:,i],norm_matrix)
  end
  norm(ncache)/norm(dcache)
end

function plot_results(
  rbinfo::RBInfo,
  feop::TransientParamFEOperator,
  fesolver::ODESolver,
  results::RBResults;
  entry=1)

  test = get_test(feop)
  trial = get_trial(feop)
  trian = get_triangulation(test)
  times = get_times(fesolver)

  name = results.name
  μ = results.params[entry]
  sol = results.sol[entry]
  sol_approx = results.sol_approx[entry]
  pointwise_err = abs.(sol-sol_approx)

  trialμt = trial(μ,times)
  fsol = FEFunction(trialμt,sol)
  fsol_approx = FEFunction(trialμt,sol_approx)
  ferr = FEFunction(trialμt,pointwise_err)

  plt_dir = joinpath(rbinfo.rb_path,"plots")
  create_dir(plt_dir)

  _plot(plt_dir,"$name",trian,fsol)
  _plot(plt_dir,"$(name)_approx",trian,fsol_approx)
  _plot(plt_dir,"$(name)_err",trian,ferr)
end

function _plot(path,name,trian,x)
  for (xt,t) in x
    writevtk(trian,path*"_$t.vtu",cellfields=[name=>xt])
  end
end
