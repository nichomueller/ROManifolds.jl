struct RBResults{T}
  name::Symbol
  params::Table
  sol::PTArray{Matrix{T}}
  sol_approx::PTArray{Matrix{T}}
  relative_err::PTArray{Vector{T}}
  fem_stats::ComputationInfo
  rb_stats::ComputationInfo

  function RBResults(
    params::Table,
    sol::PTArray{Matrix{T}},
    sol_approx::PTArray{Matrix{T}},
    fem_stats::ComputationInfo,
    rb_stats::ComputationInfo,
    args...;
    name=:vel) where T

    relative_err = compute_relative_error(sol,sol_approx,args...)
    new{T}(name,params,sol,sol_approx,relative_err,fem_stats,rb_stats)
  end
end

Base.length(r::RBResults) = length(r.params)
get_name(r::RBResults) = r.name
get_avg_error(r::RBResults) = sum(r.relative_err) / length(r)
get_speedup_time(r::RBResults) = get_avg_time(r.fem_stats) / get_avg_time(r.rb_stats)
get_speedup_memory(r::RBResults) = get_avg_nallocs(r.fem_stats) / get_avg_nallocs(r.rb_stats)

function Base.show(io::IO,r::RBResults)
  name = get_name(r)
  avg_err = get_avg_error(r)
  avg_time = get_avg_time(r.rb_stats)
  avg_nallocs = get_avg_nallocs(r.rb_stats)
  speedup_time = Float16(get_speedup_time(r)*100)
  speedup_memory = Float16(get_speedup_memory(r)*100)
  print(io,"Average online relative errors for $name: $avg_err\n")
  print(io,"Average online wall time: $avg_time [s]\n")
  print(io,"Average number of allocations: $avg_nallocs [Mb]\n")
  print(io,"FEM/RB wall time speedup: $speedup_time%\n")
  print(io,"FEM/RB memory speedup: $speedup_memory%\n")
end

function Base.show(io::IO,r::Vector{<:RBResults})
  map(r) do ri
    name = get_name(ri)
    avg_err = get_avg_error(ri)
    print(io,"Average online relative errors for $name: $avg_err\n")
  end
  r1 = first(r)
  avg_time = get_avg_time(r1.rb_stats)
  avg_nallocs = get_avg_nallocs(r1.rb_stats)
  speedup_time = Float16(get_speedup_time(r1)*100)
  speedup_memory = Float16(get_speedup_memory(r1)*100)
  print(io,"Average online wall time: $avg_time [s]\n")
  print(io,"Average number of allocations: $avg_nallocs [Mb]\n")
  print(io,"FEM/RB wall time speedup: $speedup_time%\n")
  print(io,"FEM/RB memory speedup: $speedup_memory%\n")
end

function save(info::RBInfo,r::RBResults)
  path = joinpath(info.rb_path,"results")
  save(path,r)
end

function load(info::RBInfo,T::Type{RBResults})
  path = joinpath(info.rb_path,"results")
  load(path,T)
end

function single_field_post_process(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  sol::PTArray,
  params::Table,
  sol_approx::PTArray,
  stats::NamedTuple)

  nparams = length(params)
  norm_style = info.norm_style
  norm_matrix = get_norm_matrix(info,feop,norm_style)
  _sol = space_time_matrices(sol;nparams)
  _sol_approx = space_time_matrices(sol_approx;nparams)
  fem_stats = load(info,ComputationInfo)
  rb_stats = ComputationInfo(stats,nparams)
  results = RBResults(params,_sol,_sol_approx,fem_stats,rb_stats,norm_matrix)
  show(results)
  save(info,results)
  writevtk(info,feop,fesolver,results)
  return
end

function multi_field_post_process(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  sol::PTArray,
  params::Table,
  sol_approx::PTArray,
  stats::NamedTuple)

  nblocks = length(feop.test.spaces)
  nparams = length(params)
  norm_style = info.norm_style
  offsets = field_offsets(feop.test)
  fem_stats = load(info,ComputationInfo)
  rb_stats = ComputationInfo(stats,nparams)
  blocks = map(1:nblocks) do col
    feop_col = feop[col,col]
    sol_col = get_at_offsets(sol,offsets,col)
    sol_approx_col = get_at_offsets(sol_approx,offsets,col)
    norm_matrix_col = get_norm_matrix(info,feop_col,norm_style[col])
    _sol_col = space_time_matrices(sol_col;nparams)
    _sol_approx_col = space_time_matrices(sol_approx_col;nparams)
    results = RBResults(
      params,_sol_col,_sol_approx_col,fem_stats,rb_stats,norm_matrix_col;name=Symbol("field$col"))
    save(info,results)
    writevtk(info,feop_col,fesolver,results)
    results
  end
  show(blocks)
  return
end

function allocate_cache(
  op::PTAlgebraicOperator,
  snaps::PTArray{Vector{T}}) where T

  b = allocate_residual(op,snaps)
  A = allocate_jacobian(op,snaps)

  coeff = zeros(T,1,1)
  ptcoeff = PTArray([zeros(T,1,1) for _ = eachindex(op.μ)])

  res_contrib_cache = return_cache(RBVecContributionMap(T))
  jac_contrib_cache = return_cache(RBMatContributionMap(T))

  res_cache = (b,CachedArray(coeff),CachedArray(ptcoeff)),res_contrib_cache
  jac_cache = (A,CachedArray(coeff),CachedArray(ptcoeff)),jac_contrib_cache
  res_cache,jac_cache
end

for (f,g) in zip((:single_field_rb_solver,:multi_field_rb_solver),
  (:single_field_post_process,:multi_field_post_process))
  @eval begin

    function $f(
      info::RBInfo,
      feop::PTFEOperator{Affine},
      fesolver::PODESolver,
      rbspace,
      rbres,
      rbjacs,
      snaps,
      params::Table)

      println("Solving linear RB problems")
      nsnaps_test = info.nsnaps_test
      snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
      op = get_ptoperator(fesolver,feop,snaps_test,params_test)
      rhs_cache,lhs_cache = allocate_cache(op,snaps_test)
      stats = @timed begin
        rhs = collect_rhs_contributions!(rhs_cache,info,op,rbres,rbspace)
        lhs,lhs_t = collect_lhs_contributions!(lhs_cache,info,op,rbjacs,rbspace)
        rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs+lhs_t)
      end
      approx_snaps_test = recast(rb_snaps_test,rbspace)
      $g(info,feop,fesolver,snaps_test,params_test,approx_snaps_test,stats)
    end

    function $f(
      info::RBInfo,
      feop::PTFEOperator,
      fesolver::PODESolver,
      rbspace,
      rbres,
      rbjacs,
      snaps,
      params::Table)

      println("Solving nonlinear RB problems with Newton iterations")
      nsnaps_test = info.nsnaps_test
      snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
      x = nearest_neighbor(snaps,params,params_test)
      op = get_ptoperator(fesolver,feop,snaps_test,params_test)
      rhs_cache,lhs_cache = allocate_cache(op,snaps_test)
      nl_cache = nothing
      _,conv0 = Algebra._check_convergence(fesolver.nls.ls,xrb)
      stats = @timed begin
        for iter in 1:fesolver.nls.max_nliters
          x .= recenter(x,fesolver.uh0(params_test);θ=fesolver.θ)
          xrb = space_time_projection(x,op,rbspace)
          rhs = collect_rhs_contributions!(rhs_cache,info,op,rbres,rbspace)
          lhs,lhs_t = collect_lhs_contributions!(lhs_cache,info,op,rbjacs,rbspace)
          nl_cache = rb_solve!(xrb,fesolver.nls.ls,rhs+lhs_t*xrb,lhs+lhs_t,nl_cache)
          x .+= recast(xrb,rbspace)
          op = update_ptoperator(op,x)
          isconv,conv = Algebra._check_convergence(fesolver.nls,xrb,conv0)
          println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
          if all(isconv); return; end
          if iter == nls.max_nliters
            @unreachable
          end
        end
      end
      $g(info,feop,fesolver,snaps_test,params_test,x,stats)
    end
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
  sol_approx::PTArray{Matrix{T}},
  args...) where T

  @assert length(sol) == length(sol_approx)

  time_ndofs = length(testitem(sol))
  nparams = length(sol)
  ncache = zeros(T,time_ndofs)
  dcache = zeros(T,time_ndofs)
  cache = ncache,dcache
  err = Vector{T}(undef,nparams)
  @inbounds for i = 1:nparams
    err[i] = compute_relative_error!(cache,sol[i],sol_approx[i],args...)
  end
  PTArray(err)
end

function compute_relative_error!(cache,sol,sol_approx,norm_matrix=nothing)
  ncache,dcache = cache
  @inbounds for i = axes(sol,2)
    ncache[i] = norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    dcache[i] = norm(sol[:,i],norm_matrix)
  end
  norm(ncache)/norm(dcache)
end

function Gridap.Visualization.writevtk(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  results::RBResults)

  test = get_test(feop)
  trial = get_trial(feop)
  trian = get_triangulation(test)
  times = collect(fesolver.t0:fesolver.dt:fesolver.tf-fesolver.dt)

  name = results.name
  μ = results.params[1]
  sol = results.sol[1]
  sol_approx = results.sol_approx[1]
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

function save(info::RBInfo,result::RBResults)
  path = joinpath(info.rb_path,"rbresults")
  save(path,result)
end

function load(info::RBInfo,T::Type{RBResults})
  path = joinpath(info.rb_path,"rbresults")
  load(path,T)
end
