struct RBResults
  μ::AbstractArray
  sol::AbstractArray
  sol_approx::AbstractArray
  relative_err::Float
  wall_time::Float
end

function RBResults(
  μ::AbstractArray,
  sol::AbstractArray,
  sol_approx::AbstractArray,
  wall_time::Float;
  kwargs...)

  relative_err = compute_relative_error(sol,sol_approx;kwargs...)

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors err_u: $relative_err\n";color=:red)
  printstyled("Average online wall time: $wall_time s\n";color=:red)
  printstyled("-------------------------------------------------------------\n")

  RBResults(μ,sol,sol_approx,relative_err,wall_time)
end

function Base.unique(results::Vector{RBResults})
  ntests = length(results)
  results1 = first(results)
  μ = results1.μ
  sol = results1.sol
  sol_approx = results1.sol_approx
  relative_err = sum([r.relative_err for r in results]) / ntests
  wall_time = sum([r.wall_time for r in results]) / ntests
  RBResults(μ,sol,sol_approx,relative_err,wall_time)
end

abstract type RBSolver end
struct Backslash <:RBSolver end
struct NewtonIterations <:RBSolver end

function test_rb_operator(
  info::RBInfo,
  feop::TransientFEOperator{Affine},
  rbop::TransientRBOperator{Affine},
  fesolver::ODESolver,
  rbsolver::RBSolver;
  ntests=10,
  postprocess=true)

  sols,params = load_test(info,feop,fesolver,ntests)
  norm_matrix = get_norm_matrix(info.energy_norm,feop)
  results = lazy_map((u,μ)->do_test(rbsolver,rbop,u,μ;norm_matrix),sols,params)

  if postprocess
    post_process(info,feop,fesolver,results)
  end

  return
end

function test_rb_operator(
  info::RBInfo,
  feop::TransientFEOperator,
  rbop::TransientRBOperator,
  fesolver::ODESolver,
  rbsolver::RBSolver;
  ntests=10,
  postprocess=true)

  sols,params = load_test(info,feop,fesolver,ntests)
  norm_matrix = get_norm_matrix(info.energy_norm,feop)

  results = RBResults[]
  for (u,μ) in zip(sols,params)
    ic = initial_condition(sols,params,μ)
    urb,wall_time = solve(rbsolver,rbop,μ,ic)
    push!(results,RBResults(u,urb,wall_time;norm_matrix))
  end
  if postprocess
    μ = params[1]
    r = unique(results)
    save(info,r)
    writevtk(info,feop,fesolver,r,μ)
  end
end

function load_test(
  info::RBInfo,
  feop::TransientFEOperator,
  fesolver::ODESolver,
  ntests::Int)

  try
    @check info.load_structures
    sols,params = load_test((Snapshots,Table),info)
    n = min(ntests,length(params))
    return sols[1:n],params[1:n]
  catch
    params = realization(feop,nsnaps)
    sols = collect_solutions(feop,fesolver,params)
    save_test(info,(sols,params))
    return sols,params
  end
end

function do_test(
  ::Backslash,
  rbop::TransientRBOperator{Affine},
  μ::AbstractArray,
  urb::AbstractArray;
  kwargs...)

  wall_time = @elapsed begin
    rhs = rbop.rhs(μ)
    lhs = rbop.lhs(μ)
    urb = recast(rbop,lhs \ rhs)
  end

  return RBResults(μ,u,urb,wall_time;kwargs...)
end

function do_test(
  ::NewtonIterations,
  rbop::TransientRBOperator{Affine},
  μ::AbstractArray,
  urb::AbstractArray;
  tol=1e-10,
  maxtol=1e10,
  maxit=20)

  err = 1.
  iter = 0

  wall_time = @elapsed begin
    while norm(err) ≥ tol && iter < maxit
      if norm(err) ≥ maxtol
        printstyled("Newton iterations did not converge\n";color=:red)
        return urb
      end
      rhs = rbop.rhs(μ)
      lhs = rbop.lhs(μ)
      rberr = lhs \ rhs
      err = recast(rbop,rberr)
      urb -= err
      l2_err = norm(err)/length(err)
      iter += 1
      printstyled("Newton method: ℓ^2 err = $l2_err, iter = $iter\n";color=:red)
    end
  end

  urb,wall_time
end

# function initial_condition(
#   sols::Snapshots,
#   params::Table,
#   μ::AbstractArray)

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
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  results:: Vector{RBResults})

  result = unique(results)
  save(info,result)
  writevtk(info,feop,fesolver,result)
  return
end

function Gridap.Visualization.writevtk(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
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

function load_test(T::Type{Snapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps_test")
  load(T,path)
end

function load_test(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params_test")
  load(T,path)
end

function save(info::RBInfo,result::RBResults)
  path = joinpath(info.rb_path,"rbresults")
  save(path,result)
end

function load(T::Type{RBResults},info::RBInfo)
  path = joinpath(info.rb_path,"rbresults")
  load(T,path)
end
