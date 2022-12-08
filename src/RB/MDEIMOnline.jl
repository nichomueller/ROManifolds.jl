function compute_coefficient(
  op::RBVarOperator{Affine,TT,RBSpaceSteady},
  μ::Param) where TT

  fun = get_param_function(op)
  coeff = fun(nothing,μ)
  [coeff]
end

function compute_coefficient(
  op::RBVarOperator{Affine,TT,RBSpaceUnsteady},
  μ::Param) where TT

  fun = get_param_function(op)
  timesθ = get_timesθ(op)
  coeff(tθ) = fun(nothing,μ,tθ)
  coeff.(timesθ)
end

function compute_coefficient(
  op::RBVarOperator{Top,TT,RBSpaceSteady},
  μ::Param,
  mdeim::Union{MDEIM,NTuple{2,<:MDEIM}}) where {Top,TT}

  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  red_meas = get_reduced_measure(mdeim)

  A = assemble_red_structure(op,μ,red_meas)
  mdeim_online(A,idx_lu,idx_space)
end

function compute_coefficient(
  op::RBVarOperator{Top,TT,RBSpaceUnsteady},
  μ::Param,
  mdeim::Union{MDEIM,NTuple{2,<:MDEIM}},
  info::RBInfo) where {Top,TT}

  timesθ = get_timesθ(op)
  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  red_meas = get_reduced_measure(mdeim)

  if info.st_mdeim
    idx_time = get_idx_time(mdeim)
    red_timesθ = timesθ[idx_time]
    A = assemble_red_structure(op,μ,red_meas,red_timesθ)
    mdeim_online(A,idx_lu,idx_space,get_Nt(op))
  else
    A = assemble_red_structure(op,μ,red_meas,timesθ)
    interpolate_mdeim_online(A,idx_lu,idx_space,idx_time,timesθ)
  end
end

function assemble_red_structure(
  op::RBLinOperator{Nonaffine,RBSpaceSteady},
  μ::Param,
  m::Measure)

  fun = get_fe_function(op)
  assemble_vector(v->fun(μ,m,v),get_test(op))
end

function assemble_red_structure(
  op::RBLinOperator{Nonaffine,RBSpaceUnsteady},
  μ::Param,
  m::Measure,
  timesθ::Vector{Real})

  fun = get_fe_function(op)
  v(tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))
  Matrix(v.(timesθ))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonaffine,TT,RBSpaceSteady},
  μ::Param,
  m::Measure) where TT

  fun = get_fe_function(op)
  assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonlinear,TT,RBSpaceSteady},
  μ::Param,
  m::Measure) where TT

  fun = get_fe_function(op)
  M(u) = assemble_matrix(v->fun(m,u,v),get_trial(op)(μ),get_test(op))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonaffine,TT,RBSpaceUnsteady},
  μ::Param,
  m::Measure,
  timesθ::Vector{Real}) where TT

  fun = get_fe_function(op)
  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  Matrix(M.(timesθ))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonlinear,TT,RBSpaceUnsteady},
  μ::Param,
  m::Measure,
  timesθ::Vector{Real}) where TT

  fun = get_fe_function(op)
  M(tθ,u) = assemble_matrix(v->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  M(u) = Matrix(Broadcasting(tθ -> M(tθ,u))(timesθ))
  M
end

function assemble_red_structure(
  op::RBBilinOperator{Nonaffine,TT,RBSpaceSteady},
  μ::Param,
  meas::NTuple{2,Measure}) where TT

  mmat,mlift = meas
  fun = get_fe_function(op)
  dir = get_trial(op)(μ).dirichlet_values
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M = assemble_matrix((u,v)->fun(μ,mmat,u,v),get_trial(op)(μ),get_test(op))
  Mlift = assemble_matrix((u,v)->fun(μ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = Mlift[fdofs_test,ddofs]*dir

  M,lift
end

function assemble_red_structure(
  op::RBBilinOperator{Nonlinear,TT,RBSpaceSteady},
  μ::Param,
  meas::NTuple{2,Measure}) where TT

  mmat,mlift = meas
  fun = get_fe_function(op)
  dir = get_trial(op)(μ).dirichlet_values
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(u) = assemble_matrix(v->fun(mmat,u,v),get_trial(op)(μ),get_test(op))
  Mlift(u) = assemble_matrix(v->fun(μ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(u) = Mlift(u)[fdofs_test,ddofs]*dir

  M,lift
end

function assemble_red_structure(
  op::RBBilinOperator{Nonaffine,TT,RBSpaceUnsteady},
  μ::Param,
  meas::NTuple{2,Measure},
  timesθ::Vector{Real}) where TT

  mmat,mlift = meas
  fun = get_fe_function(op)
  dir(tθ) = get_trial(op)(μ,tθ).dirichlet_values
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,mmat,u,v),get_trial(op)(μ,tθ),get_test(op))
  Mlift(tθ) = assemble_matrix((u,v)->fun(μ,tθ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ) = Mlift(tθ)[fdofs_test,ddofs]*dir(tθ)

  Matrix(M.(timesθ)),Matrix(lift.(timesθ))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonlinear,TT,RBSpaceUnsteady},
  μ::Param,
  meas::NTuple{2,Measure},
  timesθ::Vector{Real}) where TT

  mmat,mlift = meas
  fun = get_fe_function(op)
  dir(tθ) = get_trial(op)(μ,tθ).dirichlet_values
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  M(tθ,u) = assemble_matrix(v->fun(μ,tθ,mmat,u,v),get_trial(op)(μ,tθ),get_test(op))
  Mlift(tθ,u) = assemble_matrix(v->fun(μ,tθ,mlift,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(tθ,u) = Mlift(tθ,u)[fdofs_test,ddofs]*dir(tθ)
  M(u) = Matrix(Broadcasting(tθ -> M(tθ,u))(timesθ))
  lift(u) = Matrix(Broadcasting(tθ -> lift(tθ,u))(timesθ))

  M,lift
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
  lift(u) = mdeim_online(A[1](u),idx_lu[1],idx_space[1],Nt)
  M,lift
end

function interpolate_mdeim_online(
  A::AbstractArray,
  idx_lu::LU,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  timesθ::Vector{Real})

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
  timesθ::Vector{Real})

  u -> interpolate_mdeim_online(A(u),idx_lu,idx_space,idx_time,timesθ)
end

function mdeim_online(
  A::NTuple{2,T},
  idx_lu::NTuple{2,LU},
  idx_space::NTuple{2,Vector{Int}},
  Nt=1) where T

  Broadcasting((a,lu,idx)->mdeim_online(a,lu,idx,Nt))(A,idx_lu,idx_space)
end
