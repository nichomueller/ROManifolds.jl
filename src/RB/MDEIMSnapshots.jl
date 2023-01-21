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
    println("Snapshot number $k, $id")
    V(μ[k])
  end

  vals = snapshot.(eachindex(μ))
  Snapshots(id,vals)
end

function vector_snapshots(
  ::Val{true},
  op::RBLinVariable{Nonaffine},
  μ::Vector{Param})

  id = get_id(op)
  println("Building snapshots by evaluating the parametric function on the quadrature points, $id")

  Nt = get_Nt(op)
  timesθ = get_timesθ(op)

  param_vals = evaluate_param_function(op,μ)
  nred_param_vals,red_param_vals = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,red_param_vals)

  V = assemble_functional_vector(op)

  function snapshot(::RBLinVariable,k::Int)
    println("Snapshot number $k at every time, $id")
    v = Vector{Float}[]
    for nt in eachindex(timesθ)
      b = param_fun((k-1)*Nt+nt)
      push!(v,V(b))
    end
    Matrix(v)
  end

  function snapshot(::RBLiftVariable,k::Int)
    println("Snapshot number $k at every time, $id")
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
  op::RBBilinVariable{Nonaffine,<:TrialFESpace},
  μ::Vector{Param})

  id = get_id(op)
  findnz_map = get_findnz_map(op,μ)
  M = assemble_matrix(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    nonzero_values(M(μ[k]),findnz_map)
  end

  vals = snapshot.(eachindex(μ))
  findnz_map,Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinVariable{Nonaffine,Ttr},
  μ::Vector{Param}) where Ttr

  id = get_id(op)
  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    v = nonzero_values(M(μ[k]),findnz_map)
    l = lift(μ[k])
    v,l
  end

  vl = snapshot.(eachindex(μ))
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val,
  op::RBSteadyBilinVariable{Nonlinear,<:ParamTrialFESpace},
  μ::Vector{Param})

  id = get_id(op)
  bfun,bfun_lift = basis_as_fefun(op)
  findnz_map = get_findnz_map(op,bfun(1))
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(n::Int)
    println("Nonlinear snapshot number $n, $id")
    b = bfun(n)
    v = nonzero_values(M(b),findnz_map)
    v
  end

  function snapshot_lift(k::Int,n::Int)
    println("Nonlinear lift snapshot number $((k-1)*ns+n), $id")
    b = bfun_lift(μ[k],n)
    l = lift(b)
    l
  end

  ns = size(get_basis_space_col(op),2)
  nparam = min(length(μ),5)
  vals = snapshot.(1:ns)

  lifts = Vector{Float}[]
  for k = 1:nparam
    for n = 1:ns
      push!(lifts,snapshot_lift(k,n))
    end
  end

  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val,
  op::RBUnsteadyBilinVariable{Nonlinear,<:ParamTransientTrialFESpace},
  μ::Vector{Param},
  rbspaceθ::RBSpaceUnsteady)

  id = get_id(op)
  timesθ = get_timesθ(op)
  bfun,bfun_lift = basis_as_fefun(op,rbspaceθ)
  findnz_map = get_findnz_map(op,bfun(1))
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(n::Int)
    println("Nonlinear snapshot number $n, $id")
    b = bfun(n)
    v = nonzero_values(M(b),findnz_map)
    v
  end

  function snapshot_lift(k::Int,tθ::Real,n::Int)
    println("Nonlinear lift snapshot number $((k-1)*ns+n) at time $tθ, $id")
    b = bfun_lift(μ[k],tθ,n)
    l = lift(b)
    l
  end
  snapshot_lift(k::Int,n::Int) = Matrix(Broadcasting(tθ->snapshot_lift(k,tθ,n))(timesθ))

  ns = size(get_basis_space(rbspaceθ),2)
  nparam = min(length(μ),2)
  vals = snapshot.(1:ns)

  lifts = Matrix{Float}[]
  for k = 1:nparam
    for n = 1:ns
      push!(lifts,snapshot_lift(k,n))
    end
  end

  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{true},
  op::RBUnsteadyBilinVariable{Top,<:ParamTransientTrialFESpace},
  μ::Vector{Param}) where Top

  id = get_id(op)
  println("Building snapshots by evaluating the parametric function on the quadrature points, $id")

  Nt = get_Nt(op)
  timesθ = get_timesθ(op)

  param_vals = evaluate_param_function(op,μ)
  nred_param_vals,red_param_vals = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,red_param_vals)

  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_functional_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k at every time, $id")
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

function basis_as_fefun(op::RBSteadyVarOperator)
  bspace = get_basis_space_col(op)
  test = get_test(op)
  trial = get_trial(op)
  fefun(n::Int) = FEFunction(test,bspace[:,n])
  fefun_lift(μ::Param,n::Int) = FEFunction(trial(μ),bspace[:,n])
  fefun,fefun_lift
end

function basis_as_fefun(op::RBUnsteadyVarOperator,rbspaceθ::RBSpaceUnsteady)
  bspaceθ = get_basis_space(rbspaceθ)
  test = get_test(op)
  trial = get_trial(op)
  fefunθ(n::Int) = FEFunction(test,bspaceθ[:,n])
  fefunθ_lift(μ::Param,tθ::Real,n::Int) = FEFunction(trial(μ,tθ),bspaceθ[:,n])
  fefunθ,fefunθ_lift
end

function evaluate_param_function(op::RBUnsteadyVarOperator,μ::Vector{Param})
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

function reduce_param_function(op::RBUnsteadyVarOperator,vals::Matrix{Float})
  red_vals = POD(vals,Val(true))
  nred_vals = size(red_vals,2)
  red_vals_space,_ = unfold_spacetime(op,red_vals)
  nred_vals,red_vals_space
end

function interpolate_param_function(op::RBUnsteadyVarOperator,vals::Matrix{Float})
  test_quad = LagrangianQuadFESpace(get_test(op))
  param_fun = FEFunction(test_quad,vals)
  param_fun
end
