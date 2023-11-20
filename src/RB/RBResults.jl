struct RBResults{T}
  name::Symbol
  params::Table
  sol::PTArray{Matrix{T}}
  sol_approx::PTArray{Matrix{T}}
  relative_err::Vector{T}
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

function save(rbinfo::RBInfo,r::RBResults)
  path = joinpath(rbinfo.rb_path,"results")
  save(path,r)
end

function load(rbinfo::RBInfo,T::Type{RBResults})
  path = joinpath(rbinfo.rb_path,"results")
  load(path,T)
end

function post_process(
  rbinfo::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  sol::PTArray,
  params::Table,
  sol_approx::PTArray,
  stats::NamedTuple)

  nparams = length(params)
  norm_matrix = get_norm_matrix(rbinfo,feop)
  _sol = space_time_matrices(sol;nparams)
  _sol_approx = space_time_matrices(sol_approx;nparams)
  fem_stats = load(rbinfo,ComputationInfo)
  rb_stats = ComputationInfo(stats,nparams)
  results = RBResults(params,_sol,_sol_approx,fem_stats,rb_stats,norm_matrix)
  show(results)
  save(rbinfo,results)
  writevtk(rbinfo,feop,fesolver,results)
  return
end

function post_process(
  rbinfo::BlockRBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  sol::PTArray,
  params::Table,
  sol_approx::PTArray,
  stats::NamedTuple)

  nblocks = length(feop.test.spaces)
  nparams = length(params)
  offsets = field_offsets(feop.test)
  fem_stats = load(rbinfo,ComputationInfo)
  rb_stats = ComputationInfo(stats,nparams)
  blocks = map(1:nblocks) do col
    rbinfo_col = rbinfo[col]
    feop_col = feop[col,col]
    sol_col = get_at_offsets(sol,offsets,col)
    sol_approx_col = get_at_offsets(sol_approx,offsets,col)
    norm_matrix_col = get_norm_matrix(rbinfo_col,feop_col)
    _sol_col = space_time_matrices(sol_col;nparams)
    _sol_approx_col = space_time_matrices(sol_approx_col;nparams)
    results = RBResults(
      params,_sol_col,_sol_approx_col,fem_stats,rb_stats,norm_matrix_col;name=Symbol("field$col"))
    save(rbinfo,results)
    writevtk(rbinfo,feop_col,fesolver,results)
    results
  end
  show(blocks)
  return
end

function allocate_cache(op::PTOperator,snaps::PTArray{Vector{T}}) where T
  b = allocate_residual(op,snaps)
  A = allocate_jacobian(op,snaps)
  mat = CachedArray(zeros(T,1,1))
  coeff = CachedArray(zeros(T,1,1))
  ptcoeff = CachedArray(NonaffinePTArray([zeros(T,1,1) for _ = eachindex(op.μ)]))

  res_contrib_cache = return_cache(RBVecContributionMap(),snaps)
  jac_contrib_cache = return_cache(RBMatContributionMap(),snaps)

  res_cache = ((b,mat),(coeff,ptcoeff)),res_contrib_cache
  jac_cache = ((A,mat),(coeff,ptcoeff)),jac_contrib_cache
  res_cache,jac_cache
end

function rb_solver(
  rbinfo::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PODESolver,
  rbspace,
  rbres,
  rbjacs,
  snaps,
  params::Table)

  println("Solving linear RB problems")
  nsnaps_test = rbinfo.nsnaps_test
  snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
  op = get_ptoperator(fesolver,feop,snaps_test,params_test)
  cache = allocate_cache(op,snaps_test)
  stats = @timed begin
    rhs,(lhs,lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op,rbres,rbjacs,rbspace)
    rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs+lhs_t)
  end
  approx_snaps_test = recast(rb_snaps_test,rbspace)
  post_process(rbinfo,feop,fesolver,snaps_test,params_test,approx_snaps_test,stats)
end

function rb_solver(
  rbinfo::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace,
  rbres::Tuple,
  rbjacs::Tuple,
  snaps,
  params::Table)

  println("Solving nonlinear RB problems with Newton iterations")
  nsnaps_test = rbinfo.nsnaps_test
  snaps_train,params_train = snaps[1:nsnaps_test],params[1:nsnaps_test]
  snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
  x = nearest_neighbor(snaps_train,params_train,params_test)
  op = get_ptoperator(fesolver,feop,snaps_test,params_test)
  op_lin = linear_operator(op)
  op_nlin = nonlinear_operator(op)
  op_aux = auxiliary_operator(op)
  xrb = space_time_projection(x,rbspace)
  dxrb = similar(xrb)
  cache = allocate_cache(op,snaps_test)
  rhs_cache,lhs_cache = cache
  newt_cache = nothing
  conv0 = ones(nsnaps_test)
  stats = @timed begin
    rbrhs_lin,rbrhs_nlin = rbres
    rblhs_lin,rblhs_nlin,rblhs_aux = rbjacs
    rhs_lin,(lhs_lin,lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op_lin,rbrhs_lin,rblhs_lin,rbspace)
    for iter in 1:fesolver.nls.max_nliters
      x = recenter(x,fesolver.uh0(params_test);θ=fesolver.θ)
      op_nlin = update_ptoperator(op_nlin,x)
      op_aux = update_ptoperator(op_aux,x)
      rhs_nlin,(lhs_nlin,) = collect_rhs_lhs_contributions!(cache,rbinfo,op_nlin,rbrhs_nlin,rblhs_nlin,rbspace)
      lhs_aux, = collect_lhs_contributions!(lhs_cache,rbinfo,op_aux,rblhs_aux,rbspace)
      lhs = lhs_lin+lhs_t+lhs_nlin
      rhs = rhs_lin+rhs_nlin+(lhs_lin+lhs_t+lhs_aux)*xrb
      newt_cache = rb_solve!(dxrb,fesolver.nls.ls,rhs,lhs,newt_cache)
      xrb += dxrb
      x = recast(xrb,rbspace)
      isconv,conv = Algebra._check_convergence(fesolver.nls,dxrb,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
      if all(isconv); break; end
      if iter == fesolver.nls.max_nliters
        @unreachable
      end
    end
  end
  post_process(rbinfo,feop,fesolver,snaps_test,params_test,x,stats)
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

function nearest_neighbor(
  sols_train::PTArray{T},
  params_train::Table,
  params_test::Table) where T

  nparams_train = length(params_train)
  ntimes = Int(length(sols_train)/nparams_train)
  kdtree = KDTree(map(x -> SVector(Tuple(x)),params_train))
  idx_dist = map(x -> nn(kdtree,SVector(Tuple(x))),params_test)
  idx = first.(idx_dist)
  nparams_test = length(params_test)
  array = Vector{T}(undef,nparams_test*ntimes)
  @inbounds for n = 1:nparams_test
    idxn = idx[n]
    array[(n-1)*ntimes+1:n*ntimes] = sols_train[(idxn-1)*ntimes+1:idxn*ntimes]
  end
  NonaffinePTArray(array)
end

function space_time_matrices(sol::PTArray{Vector{T}};nparams=length(sol)) where T
  mat = stack(get_array(sol))
  ntimes = Int(size(mat,2)/nparams)
  array = Vector{Matrix{eltype(T)}}(undef,nparams)
  @inbounds for i = 1:nparams
    array[i] = mat[:,(i-1)*ntimes+1:i*ntimes]
  end
  NonaffinePTArray(array)
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
  err
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
  rbinfo::RBInfo,
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

  plt_dir = joinpath(rbinfo.rb_path,"plots")
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

function save(rbinfo::RBInfo,result::RBResults)
  path = joinpath(rbinfo.rb_path,"rbresults")
  save(path,result)
end

function load(rbinfo::RBInfo,T::Type{RBResults})
  path = joinpath(rbinfo.rb_path,"rbresults")
  load(path,T)
end
