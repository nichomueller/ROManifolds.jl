function mdeim_snapshots(
  info::RBInfo,
  op::RBLinVariable,
  args...)
  vector_snapshots(Val(info.fun_mdeim),op,args...)
end

function mdeim_snapshots(
  info::RBInfo,
  op::RBBilinVariable,
  args...)
  matrix_snapshots(Val(info.fun_mdeim),op,args...)
end

function vector_snapshots(
  ::Val{false},
  op::RBLinVariable{Nonaffine},
  μ::Vector{Param})

  id = get_id(op)
  V = assemble_vector(op)

  function snapshot(k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    V(μ[k])
  end

  vals = snapshot.(eachindex(μ))
  Snapshots(id,vals)
end

function vector_snapshots(
  ::Val,
  op::RBLinVariable{Nonlinear},
  μ::Vector{Param},
  uh::Snapshots)

  u_fun(k::Int) = FEFunction(op,uh[k],μ[k])
  id = get_id(op)
  V = assemble_vector(op)

  function snapshot(k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    V(μ[k],u_fun(k))
  end

  vals = snapshot.(eachindex(μ))
  Snapshots(id,vals)
end

function vector_snapshots(
  ::Val{true},
  op::RBLinVariable{Nonaffine},
  μ::Vector{Param})

  id = get_id(op)
  printstyled("Building snapshots by evaluating the parametric function
    on the quadrature points, $id \n";color=:blue)

  Nt = get_Nt(op)
  timesθ = get_timesθ(op)

  param_vals = evaluate_param_function(op,μ)
  nred_param_vals,red_param_vals = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,red_param_vals)

  V = assemble_functional_variable(op)

  function snapshot(::RBLinVariable,k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    v = Vector{Float}[]
    for nt in eachindex(timesθ)
      b = param_fun((k-1)*Nt+nt)
      push!(v,V(b))
    end
    Matrix(v)
  end

  function snapshot(::RBLiftVariable,k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    v = Vector{Float}[]
    for (nt,tθ) in enumerate(timesθ)
      b = param_fun((k-1)*Nt+nt)
      push!(v,V(b,μ[k],tθ))
    end
    Matrix(v)
  end

  vals = Broadcasting(k->snapshot(op,k))(1:nred_param_vals)
  Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinVariable,
  μ::Vector{Param})

  id = get_id(op)
  findnz_map = get_findnz_map(op,μ)
  M = assemble_matrix(op)

  function snapshot(k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    nonzero_values(M(μ[k]),findnz_map)
  end

  vals = snapshot.(eachindex(μ))
  findnz_map,Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val,
  op::RBBilinVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  uh::Snapshots) where Ttr

  u_fun(k::Int) = FEFunction(op,uh[k],μ[k])
  id = get_id(op)
  findnz_map = get_findnz_map(op,μ,u_fun)
  M = assemble_matrix(op)

  function snapshot(k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    nonzero_values(M(μ[k],u_fun(k)),findnz_map)
  end

  vals = snapshot.(eachindex(μ))
  findnz_map,Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{true},
  op::RBUnsteadyBilinVariable{Top,<:ParamTransientTrialFESpace},
  μ::Vector{Param}) where Top

  id = get_id(op)
  printstyled("Building snapshots by evaluating the parametric function
    on the quadrature points, $id \n";color=:blue)

  Nt = get_Nt(op)
  timesθ = get_timesθ(op)

  param_vals = evaluate_param_function(op,μ)
  nred_param_vals,red_param_vals = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,red_param_vals)

  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_functional_variable(op)

  function snapshot(k::Int)
    printstyled("Snapshot number $k, $id \n";color=:blue)
    v,l = Vector{Float}[],Vector{Float}[]
    for (nt,tθ) in enumerate(timesθ)
      b = param_fun((k-1)*Nt+nt)
      push!(v,nonzero_values(M(b),findnz_map))
      push!(l,lift(b,μ[k],tθ))
    end
    Matrix(v),Matrix(l)
  end

  vl = snapshot.(1:nred_param_vals)
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function evaluate_param_function(op::RBUnsteadyVariable,μ::Vector{Param})
  timesθ = get_timesθ(op)
  phys_quadp = get_phys_quad_points(op)
  param_fun = get_param_function(op)

  param(xvec::Vector{Point{D,Float}},μk::Param,tθ::Real) where D =
    Broadcasting(x->param_fun(x,μk,tθ))(xvec)[:]
  param(n::Int,μk::Param,tθ::Real) = param(phys_quadp[n],μk,tθ)
  param(μk::Param,tθ::Real) =
    Matrix(Broadcasting(n->param(n,μk,tθ))(eachindex(phys_quadp)))[:]
  param(μk::Param) = Matrix(Broadcasting(tθ->param(μk,tθ))(timesθ))[:]

  Matrix(param.(μ))
end

function reduce_param_function(op::RBUnsteadyVariable,vals::Matrix{Float})
  red_vals = POD(vals,Val(true))
  nred_vals = size(red_vals,2)
  red_vals_space,_ = unfold_spacetime(op,red_vals)
  nred_vals,red_vals_space
end

function interpolate_param_function(op::RBUnsteadyVariable,vals::Matrix{Float})
  test_quad = LagrangianQuadFESpace(get_test(op))
  param_fun = FEFunction(test_quad,vals)
  param_fun
end
