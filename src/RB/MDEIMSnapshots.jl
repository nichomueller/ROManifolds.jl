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

function mdeim_snapshots(
  op::RBLinOperator,
  ::RBInfo,
  args...)
  vector_snapshots(op,args...)
end

function mdeim_snapshots(
  op::RBBilinOperator,
  info::RBInfo,
  args...)
  matrix_snapshots(Val(info.fun_mdeim),op,args...)
end

function vector_snapshots(
  op::RBLinOperator{Nonaffine},
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
  op::RBSteadyLinOperator{Nonlinear},
  μ::Vector{Param})

  μ_mdeim = length(μ) > 5 ? μ[1:5] : μ
  nparam = length(μ_mdeim)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  ns = size(get_basis_space_col(op),2)
  V = assemble_vector(op)

  function snapshot(k::Int,n::Int)
    println("Nonlinear snapshot number $((n-1)*nparam+k), $id")
    bfun(μ_mdeim[k],n)
  end
  vals = Vector{Float}[]

  for n = 1:ns
    for k = 1:nparam
      bfun_nk = snapshot(k,n)
      push!(vals,V(bfun_nk))
    end
  end

  Snapshots(id,vals)
end

function vector_snapshots(
  op::RBUnsteadyLinOperator{Nonlinear},
  μ::Vector{Param})

  μ_mdeim = length(μ) > 2 ? μ[1:2] : μ
  nparam = length(μ_mdeim)

  id = get_id(op)
  timesθ = get_timesθ(op)
  bfun = basis_as_fefun(op)
  ns = size(get_basis_space_col(op),2)
  V(t) = assemble_vector(op,t)

  function snapshot(k::Int,tθ::Real,n::Int)
    println("Nonlinear snapshot number $((n-1)*nparam+k), $id")
    bfun(μ_mdeim[k],tθ,n)
  end

  vals = Matrix{Float}[]
  for n = 1:ns
    for k = 1:nparam
      for tθ = timesθ
        bfun_nk = snapshot(k,tθ,n)
        push!(vals,V(tθ)(bfun_nk))
      end
    end
  end

  Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{false},
  op::RBBilinOperator{Nonaffine,<:TrialFESpace},
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
  op::RBSteadyBilinOperator{Nonaffine,<:ParamTrialFESpace},
  μ::Vector{Param})

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
  ::Val{false},
  op::RBSteadyBilinOperator{Nonlinear,<:ParamTrialFESpace},
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
  ::Val{false},
  op::RBUnsteadyBilinOperator{Nonaffine,<:ParamTransientTrialFESpace},
  μ::Vector{Param})

  id = get_id(op)
  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    v = Matrix(nonzero_values(M(μ[k]),findnz_map))
    l = lift(μ[k])
    v,l
  end

  vl = snapshot.(eachindex(μ))
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{false},
  op::RBUnsteadyBilinOperator{Nonlinear,<:ParamTransientTrialFESpace},
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
  op::RBUnsteadyBilinOperator{Top,<:ParamTransientTrialFESpace},
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
  param_fefun = FEFunction(test_quad,red_vals_space)

  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    v = nonzero_values(M(param_fefun[k]),findnz_map)
    l = lift(param_fefun[k])
    v,l
  end

  vl = snapshot.(eachindex(μ))
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end
