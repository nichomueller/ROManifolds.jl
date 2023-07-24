struct RBResults
  sol::AbstractArray
  sol_approx::AbstractArray
  relative_err::Float
  online_time::Float

  function RBResults(
    sol::AbstractArray,
    sol_approx::AbstractArray,
    online_time::Float;
    kwargs...)

    relative_err = compute_relative_error(sol,sol_approx;kwargs...)

    printstyled("-------------------------------------------------------------\n")
    printstyled("Average online relative errors err_u: $relative_err\n";color=:red)
    printstyled("Average online wall time: $online_time s\n";color=:red)
    printstyled("-------------------------------------------------------------\n")

    new(sol,sol_approx,relative_err,online_time)
  end
end

abstract type RBSolver end
struct Backslash <:RBSolver end
struct NewtonIterations <:RBSolver end

for (Tfeop,Tslv,Trbop) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:RBOperator,:TransientRBOperator))

  @eval begin
    function test_rb_operator(
      info::RBInfo,
      feop::$Tfeop{Affine},
      rbop::$Trbop{Affine},
      fesolver::$Tslv,
      rbsolver::RBSolver;
      nsnaps_test=10,
      postprocess=true,
      kwargs...)

      sols,params = load_test(info,feop,fesolver,nsnaps_test)
      rb_res = Vector{RBResults}(undef,nsnaps_test)
      for (u,μ) in zip(sols,params)
        urb,wall_time = solve(rbsolver,rbop,μ)
        push!(rb_res,RBResults(u,urb,wall_time;kwargs...))
      end
      if postprocess
        save(info,rb_res)
        writevtk(info,feop,rb_res)
      end
    end

    function test_rb_operator(
      info::RBInfo,
      feop::$Tfeop,
      rbop::$Trbop,
      fesolver::$Tslv,
      rbsolver::RBSolver;
      nsnaps_test=10,
      postprocess=true,
      kwargs...)

      sols,params = load_test(info,feop,fesolver,nsnaps_test)
      rb_res = Vector{RBResults}(undef,nsnaps_test)
      for (u,μ) in zip(sols,params)
        ic = initial_condition(sols,params,μ)
        urb,wall_time = solve(rbsolver,rbop,μ,ic)
        push!(rb_res,RBResults(u,urb,wall_time;kwargs...))
      end
      if postprocess
        save(info,rb_res)
        #writevtk(info,feop,rb_res)
      end
    end

    function load_test(
      info::RBInfo,
      feop::$Tfeop,
      fesolver::$Tslv,
      nsnaps_test::Int)

      try
        sols,params = load_test((GenericSnapshots,Table),info)
        n = min(nsnaps_test,length(params))
        return sols[1:n],params[1:n]
      catch
        params = realization(feop,nsnaps_test)
        sols = collect_solutions(feop,fesolver,params;type=Matrix{Float})
        save_test(info,(sols,params))
        return sols,params
      end
    end
  end
end

function solve(
  ::Backslash,
  rbop::RBOperator{Affine},
  μ::AbstractArray,
  args...)

  online_time = @elapsed begin
    res = rbop.res(μ)
    jac = rbop.jac(μ)
    urb = recast(rbop,jac \ res)
  end
  urb,online_time
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

  online_time = @elapsed begin
    while norm(err) ≥ tol && iter < maxit
      if norm(err) ≥ maxtol
        printstyled("Newton iterations did not converge\n";color=:red)
        return urb
      end
      res = rbop.res(μ,urb)
      jac = rbop.jac(μ,urb)
      rberr = jac \ res
      err = recast(rbop,rberr)
      urb -= err
      l2_err = norm(err)/length(err)
      iter += 1
      printstyled("Newton method: ℓ^2 err = $l2_err, iter = $iter\n";color=:red)
    end
  end

  urb,online_time
end

function solve(
  ::Backslash,
  rbop::TransientRBOperator{Affine},
  μ::AbstractArray,
  args...)

  online_time = @elapsed begin
    res = rbop.res(μ)
    jac = rbop.jac(μ)
    djac = rbop.djac(μ)
    urb = recast(rbop,(jac+djac) \ res)
  end
  urb,online_time
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

  online_time = @elapsed begin
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

  urb,online_time
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

function compute_relative_error(
  sol::MultiFieldSnapshots,
  sol_approx::MultiFieldSnapshots;
  kwargs...)

  map(compute_relative_error,sol,sol_approx)
end

LinearAlgebra.norm(v::AbstractVector,::Nothing) = norm(v)

LinearAlgebra.norm(v::AbstractVector,X::AbstractMatrix) = v'*X*v

function save(info::RBInfo,rbres::RBResults)
  path = joinpath(info.rb_path,"rbresults")
  save(path,rbres)
end

function load(T::Type{RBResults},info::RBInfo)
  path = joinpath(info.rb_path,"rbresults")
  load(T,path)
end
