struct RBResults
  sol::Snapshots
  sol_approx::Snapshots
  relative_err::Float
  online_time::Float

  function RBResults(
    sol::Snapshots,
    sol_approx::Snapshots,
    online_time::Float;
    kwargs...)

    relative_err = compute_relative_error(sol,sol_approx;kwargs...)
    new(sol,sol_approx,relative_err,online_time)
  end
end

abstract type RBSolver end
struct Backslash <:RBSolver end
struct NewtonIterations <:RBSolver end

for (Top,Tslv) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver))

  @eval begin
    function test_rb_operator(
      info::RBInfo,
      feop::$Top,
      rbop::RBOperator,
      fesolver::$Tslv,
      rbsolver::RBSolver;
      nsnaps_test=10,
      postprocess=true,
      kwargs...)

      sols,params = load_test(info,feop,fesolver)
      sols_test = sols[end-nsnaps_test+1:end]
      params_test = params[end-nsnaps_test+1:end]
      res = Vector{RBResults}(undef,nsnaps_test)
      for (u,μ) in zip(sols_test,params_test)
        ic = initial_condition(sols,params,μ)
        urb,wall_time = solve(rbsolver,rbop,μ,ic)
        push!(res,RBResults(u,urb,wall_time;kwargs...))
      end
      if postprocess
        save(info,errs)
        writevtk(info,feop,errs)
      end
    end

    function load_test(
      info::RBInfo,
      feop::$Top,
      fesolver::$Tslv)

      try
        sols,params = load_test((Snapshots,Table),info)
      catch
        nsnaps = info.nsnaps_state
        params = realization(feop,nsnaps)
        sols = generate_solutions(feop,fesolver,params)
        save_test(info,(sols,params))
      end
      sols,params
    end
  end
end

function solve(
  ::Backslash,
  rbop::RBOperator{Affine},
  μ::AbstractArray,
  args...)

  t = @elapsed begin
    res = rbop.res(u,μ)
    jac = rbop.res(u,μ)
    urb = recast(rbop,jac \ res)
  end
  urb,t
end

function solve(
  ::NewtonIterations,
  rbop::RBOperator{Affine},
  μ::AbstractArray,
  urb::AbstractArray;
  tol=1e-10,
  maxtol=1e10,
  maxit=20)

  err = 1.
  iter = 0

  t = @elapsed begin
    while norm(err) ≥ tol && iter < maxit
      if norm(err) ≥ maxtol
        printstyled("Newton iterations did not converge\n";color=:red)
        return urb
      end
      res = rbop.res(urb,μ)
      jac = rbop.res(urb,μ)
      rberr = jac \ res
      err = recast(rbop,rberr)
      urb -= err
      l2_err = norm(err)/length(err)
      iter += 1
      printstyled("Newton method: ℓ^2 err = $l2_err, iter = $iter\n";color=:red)
    end
  end

  urb,t
end

function initial_condition(
  sols::Snapshots,
  params::Table,
  μ::AbstractArray)

  kdtree = KDTree(params)
  idx,dist = knn(kdtree,μ)
  get_data(sols[idx])
end

function compute_relative_error(
  sol::SingleFieldSnapshots,
  sol_approx::SingleFieldSnapshots;
  norm_matrix=nothing)

  time_ndofs = size(sol,2)
  absolute_err,snap_norm = zeros(time_ndofs),zeros(time_ndofs)
  for i = 1:Nt
    absolute_err[i] = norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    snap_norm[i] = norm(sol[:,i],norm_matrix)
  end

  norm(absolute_err)/norm(uh_norm)
end

function compute_relative_error(
  sol::MultiFieldSnapshots,
  sol_approx::MultiFieldSnapshots;
  kwargs...)

  map(compute_relative_error,sol,sol_approx)
end
