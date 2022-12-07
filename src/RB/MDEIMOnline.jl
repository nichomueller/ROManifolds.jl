function get_parameter(
  op::RBVarOperator{Affine,TT,RBSpaceSteady},
  μ::Param) where TT

  fun = get_param_function(op)
  param = fun(nothing,μ)
  [param]
end

function get_parameter(
  op::RBVarOperator{Affine,TT,RBSpaceUnsteady},
  μ::Param) where TT

  fun = get_param_function(op)
  timesθ = get_timesθ(op)
  param(tθ) = fun(nothing,μ,tθ)
  param.(timesθ)
end

function get_parameter(
  op::RBVarOperator{Top,TT,RBSpaceSteady},
  μ::Param,
  mdeim::MDEIM) where {Top,TT}

  idx_lu = get_idx_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  red_meas = get_reduced_measure(mdeim)

  A = assemble_red_structure(op,μ,red_meas)
  mdeim_online(A,idx_lu,idx_space)
end

function get_parameter(
  op::RBVarOperator{Top,TT,RBSpaceUnsteady},
  μ::Param,
  mdeim::MDEIM,
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
  op::RBLinOperator{Nonaffine,RBSpaceUnsteady},
  μ::Param,
  m::Measure,
  timesθ::Vector{Real})

  fun = get_fe_function(op)
  v(tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))
  Matrix(v.(timesθ))
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
  ::Param,
  m::Measure,
  timesθ::Vector{Real}) where TT

  fun = get_fe_function(op)
  M(tθ,u) = assemble_matrix(v->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  M(u) = Matrix(Broadcasting(tθ -> M(tθ,u))(timesθ))
  M
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

function interpolate_mdeim_online(
  A::AbstractArray,
  idx_lu::LU,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  timesθ::Vector{Real})

  red_timesθ = timesθ[idx_time]
  discarded_idx_time = setdiff(eachindex(timesθ),idx_time)
  red_param = mdeim_online(A,idx_lu,idx_space,length(idx_time))

  itp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:),red_param')
  param = zeros(length(idx_space),length(timesθ))
  param[:,idx_time] = red_param
  for it = discarded_idx_time
    param[:,it] = ScatteredInterpolation.evaluate(itp,[timesθ[it]])
  end

  param
end

function mdeim_online(
  A::Function,
  idx_lu::LU,
  idx_space::Vector{Int},
  Nt=1)

  u -> mdeim_online(A(u),idx_lu,idx_space,Nt)
end

function interpolate_mdeim_online(
  A::Function,
  idx_lu::LU,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  timesθ::Vector{Real})

  u -> interpolate_mdeim_online(A(u),idx_lu,idx_space,idx_time,timesθ)
end
