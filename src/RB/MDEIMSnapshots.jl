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
    printstyled("\n Snapshot number $k, $id";color=:blue)
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
  printstyled("\n Building snapshots by evaluating the parametric function
    on the quadrature points, $id";color=:blue)

  Nt = get_Nt(op)
  timesθ = get_timesθ(op)

  param_vals = evaluate_param_function(op,μ)
  nred_param_vals,red_param_vals = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,red_param_vals)

  V = assemble_functional_vector(op)

  function snapshot(::RBLinVariable,k::Int)
    printstyled("\n Snapshot number $k at every time, $id";color=:blue)
    v = Vector{Float}[]
    for nt in eachindex(timesθ)
      b = param_fun((k-1)*Nt+nt)
      push!(v,V(b))
    end
    Matrix(v)
  end

  function snapshot(::RBLiftVariable,k::Int)
    printstyled("\n Snapshot number $k at every time, $id";color=:blue)
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
    printstyled("\n Snapshot number $k, $id";color=:blue)
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
    printstyled("\n Snapshot number $k, $id";color=:blue)
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
  op::RBSteadyBilinVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  uh::Snapshots) where Ttr

  u_fun(k::Int) = FEFunction(get_trial(op),uh[k],μ[k])

  id = get_id(op)
  findnz_map = get_findnz_map(op,u_fun(1))
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    printstyled("\n Snapshot number $k at every time, $id";color=:blue)
    v = nonzero_values(M(u_fun(k)),findnz_map)
    l = lift(u_fun(k))
    v,l
  end

  vl = snapshot.(eachindex(μ))
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val,
  op::RBUnsteadyBilinVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  uhθ::Snapshots) where Ttr

  timesθ = get_timesθ(op)
  uhθ_fun(k::Int,n::Int) = FEFunction(get_trial(op),uhθ[k],μ[k],timesθ)(n)

  id = get_id(op)
  findnz_map = get_findnz_map(op,uhθ_fun(1,1))
  M,lift = assemble_matrix_and_lifting(op)

  function snapshot(k::Int)
    printstyled("\n Snapshot number $k at every time, $id";color=:blue)
    v,l = Vector{Float}[],Vector{Float}[]
    uk(n::Int) = uhθ_fun(k,n)
    for n in eachindex(timesθ)
      push!(v,nonzero_values(M(uk(n)),findnz_map))
      push!(l,lift(uk(n)))
    end
    Matrix(v),Matrix(l)
  end

  vl = snapshot.(eachindex(μ))
  vals,lifts = first.(vl),last.(vl)
  findnz_map,Snapshots(id,vals),Snapshots(id*:_lift,lifts)
end

function matrix_snapshots(
  ::Val{true},
  op::RBUnsteadyBilinVariable{Top,<:ParamTransientTrialFESpace},
  μ::Vector{Param}) where Top

  id = get_id(op)
  printstyled("\n Building snapshots by evaluating the parametric function
    on the quadrature points, $id";color=:blue)

  Nt = get_Nt(op)
  timesθ = get_timesθ(op)

  param_vals = evaluate_param_function(op,μ)
  nred_param_vals,red_param_vals = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,red_param_vals)

  findnz_map = get_findnz_map(op,μ)
  M,lift = assemble_functional_matrix_and_lifting(op)

  function snapshot(k::Int)
    printstyled("\n Snapshot number $k at every time, $id";color=:blue)
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

function basis_as_fefun(op::RBSteadyVariable)
  bspace = get_basis_space_col(op)
  test = get_test(op)
  trial = get_trial(op)
  fefun(n::Int) = FEFunction(test,bspace[:,n])
  fefun_lift(μ::Param,n::Int) = FEFunction(trial(μ),bspace[:,n])
  fefun,fefun_lift
end

function basis_as_fefun(op::RBUnsteadyVariable,rbspaceθ::RBSpaceUnsteady)
  bspaceθ = get_basis_space(rbspaceθ)
  test = get_test(op)
  basis_as_fefun(test,bspaceθ)
end

function basis_as_fefun(space::FESpace,basis::Matrix{Float})
  n -> FEFunction(space,basis[:,n])
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
