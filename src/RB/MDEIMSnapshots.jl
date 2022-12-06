basis_as_fefun(::RBLinOperator) = error("Not implemented")

function basis_as_fefun(
  op::RBBilinOperator{OT,<:ParamTrialFESpace,<:RBSpaceSteady}) where OT

  bspace = get_basis_space(op)
  ns = get_ns(bspace)
  trial = get_trial(op)
  fefuns(k::Int) = FEFunction(trial,bspace[:,k])
  eval.(fefuns,1:ns)
end

function basis_as_fefun(
  op::RBBilinOperator{OT,<:ParamTransientTrialFESpace,<:RBSpaceUnsteady}) where OT

  bspace = get_basis_space(op)
  ns = get_ns(bspace)
  trial = get_trial(op)
  fefuns(t::Real,k::Int) = FEFunction(trial(t),bspace[:,k])
  fefuns(t::Real) = k->fefuns(t,k)
  t -> eval.(fefuns(t),1:ns)
end

function mdeim_snapshots(op::RBLinOperator,args...)
  vector_snapshots(op,args...)
end

function mdeim_snapshots(
  op::RBBilinOperator,
  info::RBInfo,
  args...)
  matrix_snapshots(Val(info.fun_mdeim),op,args...)
end

function vector_snapshots(
  op::RBLinOperator{Nonaffine,<:RBSpaceSteady},
  μ::Vector{Param})

  id = get_id(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(μ[k])
  end

  vals = Matrix(snapshot.(eachindex(μ)))
  Snapshots(id,vals)
end

function vector_snapshots(
  op::RBLinOperator{Nonlinear,<:RBSpaceSteady},
  args...)

  id = get_id(op)
  bfun = basis_as_fefun(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(bfun[k])
  end

  vals = Matrix(snapshot.(eachindex(bfun)))
  Snapshots(id,vals)
end

function vector_snapshots(
  op::RBLinOperator{Nonaffine,<:RBSpaceUnsteady},
  μ::Vector{Param})

  id = get_id(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(μ[k],timesθ)
  end

  vals = Matrix(snapshot.(eachindex(μ)))
  Snapshots(id,vals)
end

function vector_snapshots(
  op::RBLinOperator{Nonlinear,<:RBSpaceUnsteady},
  args...)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(bfun[k],timesθ)
  end

  vals = Matrix(snapshot.(eachindex(bfun)))
  Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonaffine,<:UnconstrainedFESpace,Tsp},
  μ::Vector{Param}) where Tsp

  id = get_id(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M = assemble_matrix(op)(μ[k])
    i,v = findnz(M[:])
    i,v
  end

  iv = Matrix(snapshot.(eachindex(μ)))
  row_idx,vals = first.(iv),last.(iv)
  check_row_idx(row_idx)
  Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonlinear,<:UnconstrainedFESpace,Tsp},
  μ::Vector{Param}) where Tsp

  id = get_id(op)
  bfun = basis_as_fefun(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M = assemble_matrix(op)(bfun[k])
    i,v = findnz(M[:])
    i,v
  end

  iv = Matrix(snapshot.(eachindex(bfun)))
  row_idx,vals = first.(iv),last.(iv)
  check_row_idx(row_idx)
  Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonaffine,<:ParamTrialFESpace,<:RBSpaceSteady},
  μ::Vector{Param})

  id = get_id(op)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    i,v = findnz(M(μ[k])[:])
    i,v,lift(μ[k])
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lift = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lift)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonlinear,<:ParamTrialFESpace,<:RBSpaceSteady},
  args...)

  id = get_id(op)
  bfun = basis_as_fefun(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M,lift = assemble_matrix_and_lifting(op)(bfun[k])
    i,v = findnz(M[:])
    i,v,lift
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lift = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lift)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonaffine,<:ParamTrialFESpace,<:RBSpaceUnsteady},
  μ::Vector{Param})

  id = get_id(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M,lift = assemble_matrix_and_lifting(op)(μ[k],timesθ)
    i,v = findnz(M[:])
    i,v,lift
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lift = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lift)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonlinear,<:ParamTransientTrialFESpace,<:RBSpaceUnsteady},
  args...)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M,lift = assemble_matrix_and_lifting(op)(bfun[k],timesθ)
    i,v = findnz(M[:])
    i,v,lift
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lift = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lift)
end

function check_row_idx(row_idx::Vector{Vector{Int}})
  @assert all(Broadcasting(a->isequal(a,row_idx[1]))(row_idx)) "Need to correct snaps"
end

function matrix_snapshots(
  ::Val{true},
  op::RBBilinOperator{Top,<:ParamTransientTrialFESpace,<:RBSpaceUnsteady},
  μ::Vector{Param}) where Top

  id = get_id(op)
  println("Building snapshots by evaluating the parametric function on the quadrature points, $id")

  timesθ = get_timesθ(op)
  phys_quadp = get_phys_quad_points(op)
  ncells = length(phys_quadp)
  nquad_cell = length(phys_quadp[1])

  param_fun = get_param_function(get_background_feop(op))
  param(k,tθ,n,q) = param_fun(phys_quadp[n][q],μ[k],tθ)
  param(k,tθ,n) = Broadcasting(q -> param(k,tθ,n,q))(1:nquad_cell)
  param(k,tθ) = Matrix(Broadcasting(n -> param(k,tθ,n))(1:ncells))[:]
  param(k) = Matrix(Broadcasting(tθ -> param(k,tθ))(timesθ))[:]
  param_vals = Matrix(Broadcasting(param)(eachindex(μ)))

  red_param_vals = POD(param_vals)
  red_vals_space,_ = unfold_spacetime(op,red_param_vals)

  test_quad = LagrangianQuadFESpace(get_test(op))
  param_fefuns = FEFunction(test_quad,red_vals_space)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M,lift = assemble_matrix_and_lifting(op)(param_fefuns[k])
    i,v = findnz(M[:])
    i,v,lift
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lift = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lift)
end
