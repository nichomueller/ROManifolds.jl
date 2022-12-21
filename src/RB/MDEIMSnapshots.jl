basis_as_fefun(::RBLinOperator) = error("Not implemented")

function basis_as_fefun(
  op::RBSteadyBilinOperator{Top,<:ParamTrialFESpace}) where Top

  bspace = get_basis_space_row(op)
  trial = get_trial(op)
  fefuns(μ::Param,n::Int) = FEFunction(trial(μ),bspace[:,n])
  fefuns
end

function basis_as_fefun(
  op::RBSteadyBilinOperator{Top,<:ParamTrialFESpace}) where Top

  bspace = get_basis_space_row(op)
  trial = get_trial(op)
  fefuns(μ::Param,tθ::Real,n::Int) = FEFunction(trial(μ,tθ),bspace[:,n])
  fefuns
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
  ns = size(get_basis_space_row(op),2)
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
  ns = size(get_basis_space_row(op),2)
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
  M = assemble_matrix(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    i,v = findnz(M(μ[k])[:])
    i,v
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals = getindex.(ivl,1),getindex.(ivl,2)
  check_row_idx(row_idx)
  Snapshots(id,vals)
end

function matrix_snapshots(
  ::Val{false},
  op::RBSteadyBilinOperator{Nonaffine,<:ParamTrialFESpace},
  μ::Vector{Param})

  id = get_id(op)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    i,v = findnz(M(μ[k])[:])
    i,v,lift(μ[k])
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lifts = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{false},
  op::RBSteadyBilinOperator{Nonlinear,<:ParamTrialFESpace},
  μ::Vector{Param})

  μ_mdeim = length(μ) > 5 ? μ[1:5] : μ
  nparam = length(μ_mdeim)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  ns = size(get_basis_space_row(op),2)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int,n::Int)
    println("Nonlinear snapshot number $((n-1)*nparam+k), $id")
    bfun(μ_mdeim[k],n)
  end

  row_idx = Vector{Int}[]
  vals = Vector{Float}[]
  lifts = Vector{Float}[]
  for n = 1:ns
    for k = 1:nparam
      bfun_nk = snapshot(k,n)
      i_nk,v_nk = findnz(M(bfun_nk)[:])
      l_nk = lift(bfun_nk)
      push!(row_idx,i_nk)
      push!(vals,v_nk)
      push!(lifts,l_nk)
    end
  end

  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{false},
  op::RBUnsteadyBilinOperator{Nonaffine,<:ParamTransientTrialFESpace},
  μ::Vector{Param})

  id = get_id(op)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    iv = findnz.(M(μ[k]))
    i,v = first.(iv),last.(iv)
    check_row_idx(i)
    first(i),Matrix(v),lift(μ[k])
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lifts = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{false},
  op::RBUnsteadyBilinOperator{Nonlinear,<:ParamTrialFESpace},
  μ::Vector{Param})

  μ_mdeim = length(μ) > 2 ? μ[1:2] : μ
  nparam = length(μ_mdeim)

  id = get_id(op)
  timesθ = get_timesθ(op)
  bfun = basis_as_fefun(op)
  ns = size(get_basis_space_row(op),2)
  M_lift(t) = assemble_matrix_and_lifting(op,t)

  function snapshot(k::Int,tθ::Real,n::Int)
    println("Nonlinear snapshot number $((n-1)*nparam+k), $id")
    bfun(μ_mdeim[k],tθ,n)
  end

  row_idx = Vector{Int}[]
  vals = Matrix{Float}[]
  lifts = Matrix{Float}[]
  for n = 1:ns
    for k = 1:nparam
      for tθ = timesθ
        bfun_nk = snapshot(k,tθ,n)
        M,lift = M_lift(tθ)(bfun_nk)
        iv_nk = findnz(M)
        i_nk,v_nk = first.(iv_nk),last.(iv_nk)
        l_nk = lift
        check_row_idx(i_nk)
        push!(row_idx,first(i_nk))
        push!(vals,Matrix(v_nk))
        push!(lifts,l_nk)
      end
    end
  end

  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lifts)
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
  param_fefuns = FEFunction(test_quad,red_vals_space)

  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    i,v = findnz(M(param_fefuns[k])[:])
    i,v,lift(param_fefuns[k])
  end

  ivl = snapshot.(eachindex(μ))
  row_idx,vals,lifts = getindex.(ivl,1),getindex.(ivl,2),getindex.(ivl,3)
  check_row_idx(row_idx)
  Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function check_row_idx(row_idx::Vector{Vector{Int}})
  @assert all(Broadcasting(a->isequal(a,row_idx[1]))(row_idx)) "Need to correct snaps"
end
