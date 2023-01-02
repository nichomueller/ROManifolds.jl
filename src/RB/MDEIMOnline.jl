function compute_coefficient(
  op::Union{RBSteadyLinOperator{Affine},RBSteadyBilinOperator{Affine,Ttr},RBSteadyLiftingOperator{Affine,Ttr}},
  μ::Param) where Ttr

  fun = get_param_function(op)
  coeff = fun(nothing,μ)[1]
  [coeff]
end

function compute_coefficient(
  op::Union{RBUnsteadyLinOperator{Affine},RBUnsteadyBilinOperator{Affine,Ttr},RBUnsteadyLiftingOperator{Affine,Ttr}},
  μ::Param) where Ttr

  fun = get_param_function(op)
  timesθ = get_timesθ(op)
  coeff(tθ) = fun(nothing,μ,tθ)[1]
  Matrix(coeff.(timesθ))
end

function compute_coefficient(
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  mdeim,
  μ::Param)

  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_reduced_measure(mdeim)

  A = assemble_red_structure(op,m,μ,idx_space)
  mdeim_online(A,idx_lu)
end

function compute_coefficient(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  mdeim,
  μ::Param,
  st_mdeim=true)

  compute_coefficient(op,mdeim,μ,Val(st_mdeim))
end

function compute_coefficient(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  mdeim,
  μ::Param,
  ::Val{false})

  timesθ = get_timesθ(op)
  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_reduced_measure(mdeim)
  A = assemble_red_structure(op,m,μ,idx_space,timesθ)

  mdeim_online(A,idx_lu)
end

function compute_coefficient(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  mdeim,
  μ::Param,
  ::Val{true})

  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  idx_time = get_idx_time(mdeim)
  red_timesθ = get_reduced_timesθ(op,idx_time)
  m = get_reduced_measure(mdeim)

  A_idx_time = assemble_red_structure(op,m,μ,idx_space,red_timesθ)

  mdeim_online(A_idx_time,idx_lu)[:]
end

function assemble_red_structure(
  op::RBSteadyLinOperator{Nonaffine},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  assemble_vector(v->fun(μ,m,v),get_test(op))[idx_space]
end

function assemble_red_structure(
  op::RBUnsteadyLinOperator{Nonaffine},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  v(tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))[idx_space]
  Matrix(v.(timesθ))
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonaffine,<:TrialFESpace},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  M = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
  M[:][idx_space]
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonlinear,<:TrialFESpace},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  M(z) = assemble_matrix((u,v)->fun(m,z,u,v),get_trial(op)(μ),get_test(op))
  Midx(z) = M(z)[:][idx_space]
  Midx
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonaffine,<:TrialFESpace},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(tθ) = Vector(M(tθ)[:][idx_space])
  Matrix(Midx.(timesθ))
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonlinear,<:TrialFESpace},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  M(tθ,z) = assemble_matrix(v->fun(m,z,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(tθ,z) = Vector(M(tθ,z)[:][idx_space])
  Midx(z) = Matrix(Broadcasting(tθ->Midx(tθ,z))(timesθ))
  Midx
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonaffine,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}}) where Ttr

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M = assemble_matrix((u,v)->fun(μ,mmat,u,v),get_trial(op)(μ),get_test(op))
  Midx = Vector(M[:][idx])
  Mlift = assemble_matrix((u,v)->fun(μ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = (Mlift[fdofs_test,ddofs]*dir)[idx_lift]

  Midx,lift
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonlinear,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}}) where Ttr

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(z) = assemble_matrix((u,v)->fun(mmat,z,u,v),get_trial(op)(μ),get_test(op))
  Midx(z) = Vector(M(z)[:][idx])
  Mlift(z) = assemble_matrix((u,v)->fun(mlift,z,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(z) = (Mlift(z)[fdofs_test,ddofs]*dir)[idx_lift]

  Midx,lift
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonaffine,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}},
  timesθ::Vector{<:Real}) where Ttr

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,mmat,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(tθ) = Vector(M(tθ)[:][idx])
  Mlift(tθ) = assemble_matrix((u,v)->fun(μ,tθ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ) = (Mlift(tθ)[fdofs_test,ddofs]*dir(tθ))[idx_lift]

  Matrix(Midx.(timesθ)),Matrix(lift.(timesθ))
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonlinear,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  idx_space::NTuple{2,Vector{Int}},
  timesθ::Vector{<:Real}) where Ttr

  mmat,mlift = m_mat_lift
  idx,idx_lift = idx_space

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(tθ,z) = assemble_matrix(v->fun(mmat,z,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(tθ,z) = Vector(M(tθ,z)[:][idx])
  Midx(z) = Matrix(Broadcasting(tθ->Midx(tθ,z))(timesθ))
  Mlift(z) = assemble_matrix((u,v)->fun(mlift,z,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ,z) = (Mlift(z)[fdofs_test,ddofs]*dir(tθ))[idx_lift]
  lift(z) = Matrix.(Broadcasting(tθ -> lift(tθ,z))(timesθ))

  Midx,lift
end

#= function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonaffine,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int}) where Ttr

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = (Mlift[fdofs_test,ddofs]*dir)[idx_space]

  lift
end =#

function assemble_red_structure(
  op::RBSteadyLiftingOperator,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  assemble_red_lifting(op,m,μ,idx_space)
end

function assemble_red_structure(
  op::RBUnsteadyLiftingOperator,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  assemble_red_lifting(op,m,μ,idx_space,timesθ)
end

function assemble_red_lifting(
  op,
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
  op,
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
  idx_lu::LU)

  solve_lu(Aidx,idx_lu)
end

function mdeim_online(
  A::Function,
  idx_lu::LU)

  u -> mdeim_online(A(u),idx_lu)
end

function mdeim_online(
  A::NTuple{2,AbstractArray},
  idx_lu::NTuple{2,LU})

  M = mdeim_online(A[1],idx_lu[1])
  lift = mdeim_online(A[2],idx_lu[2])
  M,lift
end

function mdeim_online(
  A::NTuple{2,Function},
  idx_lu::NTuple{2,LU})

  M(u) = mdeim_online(A[1](u),idx_lu[1])
  lift(u) = mdeim_online(A[2](u),idx_lu[2])
  M,lift
end

function interpolate_mdeim_online(
  A::AbstractArray,
  idx_lu::LU,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  timesθ::Vector{<:Real})

  red_timesθ = timesθ[idx_time]
  discarded_idx_time = setdiff(eachindex(timesθ),idx_time)
  red_coeff = mdeim_online(A,idx_lu)

  itp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:),red_coeff')
  coeff = zeros(length(idx_space),length(timesθ))
  coeff[:,idx_time] = red_coeff
  for it = discarded_idx_time
    coeff[:,it] = ScatteredInterpolation.evaluate(itp,[timesθ[it]])
  end

  coeff
end

function interpolate_mdeim_online(
  A::Function,
  idx_lu::LU,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  timesθ::Vector{<:Real})

  u -> interpolate_mdeim_online(A(u),idx_lu,idx_space,idx_time,timesθ)
end
