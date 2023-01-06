function basis_as_fefun(op::RBVarOperator)
  bspace = get_basis_space_row(op)
  test = get_test(op)
  fefuns(n::Int) = FEFunction(test,bspace[:,n])
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
  args...)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  findnz_map = get_findnz_map(op,bfun(1))
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(n::Int)
    println("Nonlinear snapshot number $n, $id")
    b = bfun(n)
    v = nonzero_values(M(b),findnz_map)
    l = lift(b)
    v,l
  end

  ns = size(get_basis_space_row(op),2)
  vl = snapshot.(1:ns)
  vals,lifts = first.(vl),last.(vl)
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
  op::RBBilinOperator{Nonlinear,Ttr},
  args...) where Ttr

  id = get_id(op)
  bfun = basis_as_fefun(op)
  findnz_map = get_findnz_map(op,bfun(1))
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(n::Int)
    println("Nonlinear snapshot number $n, $id")
    b = bfun(n)
    v = Matrix(nonzero_values(M(b),findnz_map))
    l = lift(b)
    v,l
  end

  ns = size(get_basis_space_row(op),2)
  vl = snapshot.(1:ns)
  vals,lifts = first.(vl),last.(vl)
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
  param_fefuns = FEFunction(test_quad,red_vals_space)

  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    v = nonzero_values(M(param_fefuns[k]),findnz_map)
    l = lift(param_fefuns[k])
    v,l
  end

  vl = snapshot.(eachindex(μ))
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end
