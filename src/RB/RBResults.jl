abstract type AbstractRBResults end

Base.length(r::AbstractRBResults) = length(r.params)

struct RBResults <: AbstractRBResults
  params::Table
  sol::PTArray
  sol_approx::PTArray
  relative_err::Vector{Float}
  wall_time::Float
end

function RBResults(
  params::Table,
  sol::PTArray,
  sol_approx::PTArray,
  wall_time::Float;
  kwargs...)

  relative_err = compute_relative_error(sol,sol_approx;kwargs...)
  new(params,sol,sol_approx,relative_err,wall_time)
end

get_avg_error(r::RBResults) = sum(r.relative_err) / length(r)
get_avg_time(r::RBResults) = r.wall_time / length(r)

function Base.show(io::IO,r::RBResults)
  avg_err = get_avg_error(r)
  avg_time = get_avg_time(r)
  print(io,"-------------------------------------------------------------\n")
  print(io,"Average online relative errors: $avg_err\n")
  print(io,"Average online wall time: $avg_time s\n")
  print(io,"-------------------------------------------------------------\n")
end

function Base.first(r::RBResults)
  μ = r.params[1]
  sol = r.sol[1]
  sol_approx = r.sol_approx[1]
  relative_err = get_avg_error(r)
  wall_time = get_avg_time(r)
  RBResults(μ,sol,sol_approx,relative_err,wall_time)
end

function allocate_sys_cache(
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::AbstractRBSpace{T},
  params::Table) where T

  times = get_times(fesolver)
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,params,times)
  b = allocate_residual(ode_op,sols_test,ode_cache)
  A = allocate_jacobian(ode_op,sols_test,ode_cache)

  rb_ndofs = num_rb_dofs(rbspace)
  n = length(params)*length(times)
  brb = PTArray([zeros(T,rb_ndofs) for _ = 1:n])
  Arb = PTArray([zeros(T,rb_ndofs,rb_ndofs) for _ = 1:n])

  k = RBContributionMap()
  rbres = testvalue(RBAffineDecomposition{T},feop;vector=true)
  rbjac = testvalue(RBAffineDecomposition{T},feop;vector=false)
  res_contrib_cache = return_cache(k,rbres.basis_space,rbres.basis_time)
  jac_contrib_cache = return_cache(k,rbjac.basis_space,rbjac.basis_time)

  res_cache = (CachedArray(b),CachedArray(brb[1]),CachedArray(brb)),res_contrib_cache
  jac_cache = (CachedArray(A),CachedArray(Arb[1]),CachedArray(Arb)),jac_contrib_cache
  res_cache,jac_cache
end

function test_rb_solver(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PODESolver,
  rbspace::AbstractRBSpace,
  rbres::AbstractRBAlgebraicContribution,
  rbjacs::AbstractRBAlgebraicContribution,
  ntests::Int)

  snaps_test,params_test = load_test(info,feop,fesolver,ntests)

  printstyled("Solving linear RB problems\n";color=:blue)
  rhs_cache,lhs_cache = allocate_sys_cache(feop,fesolver,rbspace,params_test)
  rhs = collect_rhs_contributions(rhs_cache,info,feop,fesolver,rbres,params_test)
  lhs = collect_lhs_contributions(lhs_cache,info,feop,fesolver,rbjacs,params_test)

  wall_time = @elapsed begin
    rb_snaps_test = solve(fesolver,rbspace,rhs,lhs)
  end
  approx_snaps_test = recast(rbspace,rb_snaps_test)
  RBResults(info,feop,snaps_test,approx_snaps_test,wall_time)
end

function test_rb_solver(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::AbstractRBSpace,
  rbres::AbstractRBAlgebraicContribution,
  rbjacs::AbstractRBAlgebraicContribution,
  ntests::Int)

  snaps_test,params_test = load_test(info,feop,fesolver,ntests)

  printstyled("Solving nonlinear RB problems with Newton iterations\n";color=:blue)
  rhs_cache,lhs_cache = allocate_sys_cache(feop,fesolver,rbspace,params_test)
  nl_cache = nothing
  x = initial_guess(info,params_test)
  _,conv0 = Algebra._check_convergence(fesolver.nls,x)
  wall_time = @elapsed begin
    for iter in 1:fesolver.nls.max_nliters
      rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbres,params_test,x)
      lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rbjacs,params_test,x)
      nl_cache = solve!(x,fesolver,rhs,lhs,nl_cache)
      x .-= recast(rbspace,x)
      isconv,conv = Algebra._check_convergence(fesolver.nls,x,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv))) \n")
      if all(isconv); return; end
      if iter == nls.max_nliters
        @unreachable
      end
    end
  end
  RBResults(info,feop,snaps_test,x,wall_time)
end

function load_test(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  ntests::Int)

  try
    @check info.load_structures
    sols,params = load_test((Snapshots,Table),info)
    return sols[1:ntests],params[1:ntests]
  catch
    params = realization(feop,ntests)
    sols = collect_solutions(fesolver,feop,params)
    save_test(info,(sols,params))
    return sols[1:ntests],params[1:ntests]
  end
end

function Algebra.solve(fesolver::PODESolver,rhs::PTArray,lhs::PTArray)
  x = copy(rhs)
  cache = nothing
  solve!(x,fesolver,rhs,lhs,cache)
end

function Algebra.solve!(
  x::PTArray,
  fesolver::PODESolver,
  rhs::PTArray,
  lhs::PTArray,
  ::Nothing)

  lhsaff,rhsaff = Nonaffine(),Nonaffine()
  ss = symbolic_setup(fesolver.nls.ls,testitem(lhs))
  ns = numerical_setup(ss,lhs,lhsaff)
  _loop_solve!(x,ns,rhs,lhsaff,rhsaff)
end

# function initial_condition(
#   info::RBInfo,
    # params_test::Table)
    # snaps,params = load(info,{Snapshots,Table})
#   kdtree = KDTree(params)
#   idx,dist = knn(kdtree,μ)
#   get_data(sols[idx])
# end

function compute_relative_error(
  sol::AbstractVector,
  sol_approx::AbstractVector;
  norm_matrix=nothing)

  absolute_err = norm(sol-sol_approx,norm_matrix)
  snap_norm = norm(sol,norm_matrix)
  absolute_err/snap_norm
end

function compute_relative_error(
  sol::AbstractMatrix,
  sol_approx::AbstractMatrix;
  norm_matrix=nothing)

  time_ndofs = size(sol,2)
  absolute_err,snap_norm = zeros(time_ndofs),zeros(time_ndofs)
  for i = 1:time_ndofs
    absolute_err[i] = norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    snap_norm[i] = norm(sol[:,i],norm_matrix)
  end

  norm(absolute_err)/norm(snap_norm)
end

LinearAlgebra.norm(v::AbstractVector,::Nothing) = norm(v)

LinearAlgebra.norm(v::AbstractVector,X::AbstractMatrix) = v'*X*v

function post_process(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  results:: Vector{RBResults})

  result = unique(results)
  save(info,result)
  writevtk(info,feop,fesolver,result)
  return
end

function Gridap.Visualization.writevtk(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  result::RBResults)

  μ = result.μ
  test = get_test(feop)
  trial = get_trial(feop)
  trian = get_triangulation(test)
  times = get_times(fesolver)

  sol = result.sol
  sol_approx = result.sol_approx
  pointwise_err = abs.(sol-sol_approx)

  plt_dir = joinpath(info.rb_path,"plots")
  create_dir!(plt_dir)
  for (it,t) in enumerate(times)
    fsol = FEFunction(trial(μ,t),sol[:,it])
    fsol_approx = FEFunction(trial(μ,t),sol_approx[:,it])
    ferr = FEFunction(trial(μ,t),pointwise_err[:,it])
    writevtk(trian,joinpath(plt_dir,"sol_$(it).vtu"),cellfields=["err"=>fsol])
    writevtk(trian,joinpath(plt_dir,"sol_approx_$(it).vtu"),cellfields=["err"=>fsol_approx])
    writevtk(trian,joinpath(plt_dir,"err_$(it).vtu"),cellfields=["err"=>ferr])
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
