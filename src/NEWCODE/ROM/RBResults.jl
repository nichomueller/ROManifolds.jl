struct RBResults
  sol::AbstractArray
  sol_approx::AbstractArray
  relative_err::Float
  wall_time::Float
end

function RBResults(
  sol::AbstractArray,
  sol_approx::AbstractArray,
  wall_time::Float;
  kwargs...)

  relative_err = compute_relative_error(sol,sol_approx;kwargs...)

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors err_u: $relative_err\n";color=:red)
  printstyled("Average online wall time: $wall_time s\n";color=:red)
  printstyled("-------------------------------------------------------------\n")

  RBResults(sol,sol_approx,relative_err,wall_time)
end

function Base.unique(rb_res::Vector{RBResults})
  ntests = length(rb_res)
  rb_res1 = first(rb_res)
  sol = rb_res1.sol
  sol_approx = rb_res1.sol_approx
  relative_err = sum([r.relative_err for r in rb_res]) / ntests
  wall_time = sum([r.wall_time for r in rb_res]) / ntests
  RBResults(sol,sol_approx,relative_err,wall_time)
end

abstract type RBSolver end
struct Backslash <:RBSolver end
struct NewtonIterations <:RBSolver end

function test_rb_operator(
  info::RBInfo,
  feop::ODEOperator{Affine},
  rbop::TransientRBOperator{Affine},
  fesolver::ODESolver,
  rbsolver::RBSolver;
  ntests=10,
  postprocess=true,
  kwargs...)

  sols,params = load_test(info,feop,fesolver,ntests)
  rb_res = RBResults[]
  for (u,μ) in zip(sols,params)
    urb,wall_time = solve(rbsolver,rbop,μ)
    push!(rb_res,RBResults(u,urb,wall_time;kwargs...))
  end
  if postprocess
    μ = params[1]
    r = unique(rb_res)
    save(info,r)
    writevtk(info,feop,fesolver,r,μ)
  end
end

function test_rb_operator(
  info::RBInfo,
  feop::ODEOperator,
  rbop::TransientRBOperator,
  fesolver::ODESolver,
  rbsolver::RBSolver;
  ntests=10,
  postprocess=true,
  kwargs...)

  sols,params = load_test(info,feop,fesolver,ntests)
  rb_res = RBResults[]
  for (u,μ) in zip(sols,params)
    ic = initial_condition(sols,params,μ)
    urb,wall_time = solve(rbsolver,rbop,μ,ic)
    push!(rb_res,RBResults(u,urb,wall_time;kwargs...))
  end
  if postprocess
    μ = params[1]
    r = unique(rb_res)
    save(info,r)
    writevtk(info,feop,fesolver,r,μ)
  end
end

function load_test(
  info::RBInfo,
  feop::ODEOperator,
  fesolver::ODESolver,
  ntests::Int)

  try
    @check info.load_structures
    sols,params = load_test((Snapshots,Table),info)
    n = min(ntests,length(params))
    return sols[1:n],params[1:n]
  catch
    params = realization(feop,ntests)
    sols = collect_solutions(feop,fesolver,params;type=Matrix{Float})
    save_test(info,(sols,params))
    return sols,params
  end
end

function solve(
  ::Backslash,
  rbop::TransientRBOperator{Affine},
  μ::AbstractArray,
  args...)

  wall_time = @elapsed begin
    res = rbop.res(μ)
    jac = rbop.jac(μ)
    djac = rbop.djac(μ)
    urb = recast(rbop,(jac+djac) \ res)
  end
  urb,wall_time
end

function solve(
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
      res = rbop.res(μ,urb)
      jac = rbop.jac(μ,urb)
      djac = rbop.djac(μ)
      rberr = (jac+djac) \ res
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

function Gridap.Visualization.writevtk(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rb_res::RBResults,
  μ::AbstractArray)

  test = get_test(feop)
  trial = get_trial(feop)
  trian = get_triangulation(test)
  times = get_times(fesolver)

  sol = rb_res.sol
  sol_approx = rb_res.sol_approx
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

function save(info::RBInfo,rb_res::RBResults)
  path = joinpath(info.rb_path,"rbresults")
  save(path,rb_res)
end

function load(T::Type{RBResults},info::RBInfo)
  path = joinpath(info.rb_path,"rbresults")
  load(T,path)
end
