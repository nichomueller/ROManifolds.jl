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
  Matrix(coeff.(timesθ)')
end

function compute_coefficient(
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  mdeim,
  μ::Param)

  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_reduced_measure(mdeim)

  A = assemble_red_structure(op,m,μ)
  mdeim_online(A,idx_lu,idx_space)
end

function compute_coefficient(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  mdeim,
  μ::Param,
  st_mdeim=true)

  timesθ = get_timesθ(op)
  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_reduced_measure(mdeim)

  if !st_mdeim
    idx_time = get_idx_time(mdeim)
    red_timesθ = timesθ[idx_time]
    A = assemble_red_structure(op,m,μ,red_timesθ)
    interpolate_mdeim_online(A,idx_lu,idx_space,idx_time,timesθ)
  else
    A = assemble_red_structure(op,m,μ,timesθ)
    mdeim_online(A,idx_lu,idx_space,get_Nt(op))
  end
end

function assemble_red_structure(
  op::RBSteadyLinOperator{Nonaffine},
  m::Measure,
  μ::Param)

  fun = get_fe_function(op)
  assemble_vector(v->fun(μ,m,v),get_test(op))
end

function assemble_red_structure(
  op::RBUnsteadyLinOperator{Nonaffine},
  m::Measure,
  μ::Param,
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  v(tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))
  Matrix(v.(timesθ))
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonaffine,<:TrialFESpace},
  m::Measure,
  μ::Param)

  fun = get_fe_function(op)
  assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonlinear,<:TrialFESpace},
  m::Measure,
  μ::Param)

  fun = get_fe_function(op)
  z -> assemble_matrix((u,v)->fun(m,z,u,v),get_trial(op)(μ),get_test(op))
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonaffine,<:TrialFESpace},
  m::Measure,
  μ::Param,
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  Matrix(M.(timesθ))
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonlinear,<:TrialFESpace},
  m::Measure,
  μ::Param,
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  tθ = rand(timesθ)
  M(z) = assemble_matrix(v->fun(m,z,u,v),get_trial(op)(μ,tθ),get_test(op))
  M
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonaffine,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param) where Ttr

  mmat,mlift = m_mat_lift

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M = assemble_matrix((u,v)->fun(μ,mmat,u,v),get_trial(op)(μ),get_test(op))
  Mlift = assemble_matrix((u,v)->fun(μ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = Mlift[fdofs_test,ddofs]*dir

  M,lift
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonlinear,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param) where Ttr

  mmat,mlift = m_mat_lift

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(z) = assemble_matrix((u,v)->fun(mmat,z,u,v),get_trial(op)(μ),get_test(op))
  Mlift(z) = assemble_matrix((u,v)->fun(mlift,z,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(z) = Mlift(z)[fdofs_test,ddofs]*dir

  M,lift
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonaffine,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  timesθ::Vector{<:Real}) where Ttr

  mmat,mlift = m_mat_lift

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,mmat,u,v),get_trial(op)(μ,tθ),get_test(op))
  Mlift(tθ) = assemble_matrix((u,v)->fun(μ,tθ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ) = Mlift(tθ)[fdofs_test,ddofs]*dir(tθ)

  Matrix(M.(timesθ)),Matrix(lift.(timesθ))
end

function assemble_red_structure(
  op::RBUnsteadyBilinOperator{Nonlinear,Ttr},
  m_mat_lift::NTuple{2,Measure},
  μ::Param,
  timesθ::Vector{<:Real}) where Ttr

  mmat,mlift = m_mat_lift

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(z) = assemble_matrix((u,v)->fun(mmat,z,u,v),get_trial(op)(μ,rand(timesθ)),get_test(op))
  Mlift(z) = assemble_matrix((u,v)->fun(mlift,z,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ,u) = Mlift(u)[fdofs_test,ddofs]*dir(tθ)
  lift(u) = Matrix.(Broadcasting(tθ -> lift(tθ,u))(timesθ))

  M,lift
end

function assemble_red_structure(
  op::RBSteadyBilinOperator{Nonaffine,Ttr},
  m::Measure,
  μ::Param) where Ttr

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = Mlift[fdofs_test,ddofs]*dir

  lift
end

function assemble_red_structure(
  op::RBSteadyLiftingOperator,
  m::Measure,
  μ::Param)

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = Mlift[fdofs_test,ddofs]*dir

  lift
end

function assemble_red_structure(
  op::RBUnsteadyLiftingOperator,
  m::Measure,
  μ::Param,
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  Mlift(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ) = Mlift(tθ)[fdofs_test,ddofs]*dir(tθ)

  Matrix(lift.(timesθ))
end

function solve_lu(A::AbstractArray,lu::LU)
  P_A = lu.P*A
  y = lu.L \ P_A
  x = lu.U \ y
  x
end

function mdeim_online(
  A::AbstractArray,
  idx_lu::LU,
  idx_space::Vector{Int},
  Nt=1)

  if Nt > 1
    Aidx = Matrix(reshape(A,:,Nt)[idx_space,:])
  else
    Aidx = A[:][idx_space]
  end
  solve_lu(Aidx,idx_lu)
end

function mdeim_online(
  A::Function,
  idx_lu::LU,
  idx_space::Vector{Int},
  Nt=1)

  u -> mdeim_online(A(u),idx_lu,idx_space,Nt)
end

function mdeim_online(
  A::NTuple{2,AbstractArray},
  idx_lu::NTuple{2,LU},
  idx_space::NTuple{2,Vector{Int}},
  Nt=1)

  M = mdeim_online(A[1],idx_lu[1],idx_space[1],Nt)
  lift = mdeim_online(A[2],idx_lu[2],idx_space[2],Nt)
  M,lift
end

function mdeim_online(
  A::NTuple{2,Function},
  idx_lu::NTuple{2,LU},
  idx_space::NTuple{2,Vector{Int}},
  Nt=1)

  M(u) = mdeim_online(A[1](u),idx_lu[1],idx_space[1],Nt)
  lift(u) = mdeim_online(A[2](u),idx_lu[2],idx_space[2],Nt)
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
  red_coeff = mdeim_online(A,idx_lu,idx_space,length(idx_time))

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

#= function mdeim_online(
  A::NTuple{2,T},
  idx_lu::NTuple{2,LU},
  idx_space::NTuple{2,Vector{Int}},
  Nt=1) where T

  Broadcasting((a,lu,idx)->mdeim_online(a,lu,idx,Nt))(A,idx_lu,idx_space)
end =#
