function compute_coefficient(
  op::RBSteadyVarOperator{Affine,Ttr},
  μ::Param) where Ttr

  fun = get_param_function(op)
  coeff = fun(nothing,μ)[1]
  [coeff]
end

function compute_coefficient(
  op::RBUnsteadyVarOperator{Affine,Ttr},
  μ::Param) where Ttr

  fun = get_param_function(op)
  timesθ = get_timesθ(op)
  coeff(tθ) = fun(nothing,μ,tθ)[1]
  Matrix(coeff.(timesθ))
end

function compute_coefficient(
  op::RBSteadyVarOperator,
  mdeim,
  μ::Param)

  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)

  A = assemble_red_structure(op,m,μ,idx_space)
  mdeim_online(A,red_lu)
end

function compute_coefficient(
  op::RBUnsteadyVarOperator,
  mdeim,
  μ::Param,
  st_mdeim=false)

  compute_coefficient(op,mdeim,μ,Val(st_mdeim))
end

function compute_coefficient(
  op::RBUnsteadyVarOperator,
  mdeim,
  μ::Param,
  ::Val{false})

  timesθ = get_timesθ(op)
  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = assemble_red_structure(op,m,μ,idx_space,timesθ)

  mdeim_online(A,red_lu)
end

function compute_coefficient(
  op::RBUnsteadyVarOperator,
  mdeim,
  μ::Param,
  ::Val{true})

  red_lu = get_red_lu_factors(mdeim)
  idx_space,idx_time = get_idx_space(mdeim),get_idx_time(mdeim)
  red_timesθ = get_reduced_timesθ(op,idx_time)
  m = get_red_measure(mdeim)

  A_idx = assemble_red_structure(op,m,μ,idx_space,red_timesθ)
  A_st = spacetime_vector(A_idx)
  coeff = mdeim_online(A_st,red_lu)
  interp_coeff_time(mdeim,coeff)
end

function assemble_red_structure(
  op::RBSteadyLinVariable{Nonaffine},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  assemble_vector(v->fun(μ,m,v),get_test(op))[idx_space]
end

function assemble_red_structure(
  op::RBUnsteadyLinVariable{Nonaffine},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  v(tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))[idx_space]
  Matrix(v.(timesθ))
end

function assemble_red_structure(
  op::RBSteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int}) where Ttr

  fun = get_fe_function(op)
  M = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
  Vector(M[:][idx_space])
end

function assemble_red_structure(
  op::RBSteadyBilinVariable{Nonlinear,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int}) where Ttr

  fun = get_fe_function(op)
  M(z) = assemble_matrix((u,v)->fun(m,z,u,v),get_trial(op)(μ),get_test(op))
  Midx(z) = Vector(M(z)[:][idx_space])
  Midx
end

function assemble_red_structure(
  op::RBUnsteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real}) where Ttr

  fun = get_fe_function(op)
  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(tθ) = Vector(M(tθ)[:][idx_space])
  Matrix(Midx.(timesθ))
end

function assemble_red_structure(
  op::RBUnsteadyBilinVariable{Nonlinear,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real}) where Ttr

  fun = get_fe_function(op)
  M(tθ,z) = assemble_matrix((u,v)->fun(m,z,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(tθ,z) = Vector(M(tθ,z(tθ))[:][idx_space])
  Midx(z) = Matrix(Broadcasting(tθ->Midx(tθ,z))(timesθ))
  Midx
end

function assemble_red_structure(
  op::RBSteadyBilinVariable,
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}})

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space
  Midx = assemble_red_structure(op,mmat,μ,idx)
  lift = assemble_red_lifting(op,mlift,μ,idx_lift)

  Midx,lift
end

function assemble_red_structure(
  op::RBUnsteadyBilinVariable,
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}},
  timesθ::Vector{<:Real})

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space
  Midx = assemble_red_structure(op,mmat,μ,idx,timesθ)
  lift = assemble_red_lifting(op,mlift,μ,idx_lift,timesθ)

  Midx,lift
end

function assemble_red_structure(
  op::RBUnsteadyBilinVariable,
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}},
  timesθ::NTuple{2,Vector{<:Real}})

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space
  tθ,tθ_lift = timesθ
  Midx = assemble_red_structure(op,mmat,μ,idx,tθ)
  lift = assemble_red_lifting(op,mlift,μ,idx_lift,tθ_lift)

  Midx,lift
end

function assemble_red_structure(
  op::RBSteadyLiftVariable,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  assemble_red_lifting(op,m,μ,idx_space)
end

function assemble_red_structure(
  op::RBUnsteadyLiftVariable,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  assemble_red_lifting(op,m,μ,idx_space,timesθ)
end

function assemble_red_lifting(
  op::RBSteadyVarOperator,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = (Mlift[fdofs_test,ddofs]*dir)[idx_space]

  lift
end

function assemble_red_lifting(
  op::RBSteadyVarOperator{Nonlinear,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int}) where Ttr

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift(z) = assemble_matrix((u,v)->fun(m,z,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(z) = (Mlift(z)[fdofs_test,ddofs]*dir)[idx_space]

  lift
end

function assemble_red_lifting(
  op::RBUnsteadyVarOperator,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ) = (Mlift(tθ)[fdofs_test,ddofs]*dir(tθ))[idx_space]

  Matrix(lift.(timesθ))
end

function assemble_red_lifting(
  op::RBUnsteadyVarOperator{Nonlinear,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real}) where Ttr

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift(z) = assemble_matrix((u,v)->fun(m,z,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ,z) = (Mlift(z(tθ))[fdofs_test,ddofs]*dir(tθ))[idx_space]
  lift(z) = Matrix(Broadcasting(tθ -> lift(tθ,z))(timesθ))

  lift
end

function solve_lu(A::Vector{Float},lu::LU)
  P_A = lu.P*A
  y = lu.L \ P_A
  x = lu.U \ y
  x
end

function solve_lu(A::Matrix{Float},lu::LU)
  P_A = lu.P*A
  y = lu.L \ P_A
  x = lu.U \ y
  Matrix(x')
end

function mdeim_online(
  Aidx::AbstractArray,
  red_lu::LU)

  solve_lu(Aidx,red_lu)
end

function mdeim_online(
  A::Function,
  red_lu::LU)

  u -> mdeim_online(A(u),red_lu)
end

function mdeim_online(
  A::NTuple{2,AbstractArray},
  red_lu::NTuple{2,LU})

  M = mdeim_online(A[1],red_lu[1])
  lift = mdeim_online(A[2],red_lu[2])
  M,lift
end

function mdeim_online(
  A::NTuple{2,Function},
  red_lu::NTuple{2,LU})

  M(u) = mdeim_online(A[1](u),red_lu[1])
  lift(u) = mdeim_online(A[2](u),red_lu[2])
  M,lift
end

function interp_coeff_time(
  mdeim::MDEIMUnsteady,
  coeff::AbstractVector)

  bs = get_basis_space(mdeim)
  bt = get_basis_time(mdeim)
  Qs = size(bs,2)
  Qt = size(bt,2)
  sorted_idx(qs) = [(i-1)*Qs+qs for i=1:Qt]
  bt_times_coeff(qs) = bt*coeff[sorted_idx(qs)]
  Matrix(bt_times_coeff.(1:Qs))
end

function interp_coeff_time(
  mdeim::MDEIMUnsteady,
  coeff::Function)

  u -> interp_coeff_time(mdeim,coeff(u))
end

function interp_coeff_time(
  mdeim::NTuple{2,MDEIMUnsteady},
  coeff::NTuple{2,Any})

  interp_coeff_time.(mdeim,coeff)
end
