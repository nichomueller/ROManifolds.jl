function get_parameter(
  op::RBVarOperator{Affine,TT,RBSpaceSteady},
  μ::Param,
  args...) where TT

  fun = get_param_function(op)
  param = fun(μ)
  param
end

function get_parameter(
  op::RBVarOperator{Affine,TT,RBSpaceUnsteady},
  μ::Param,
  args...) where TT

  fun = get_param_function(op)
  timesθ = get_timesθ(op)
  param(tθ) = fun(μ,tθ)
  param.(timesθ)
end

function get_parameter(
  op::RBVarOperator{Top,TT,RBSpaceSteady},
  μ::Param,
  mdeim::MDEIM,
  args...) where {Top,TT}

  meas,field,_ = args
  bs = get_basis_space(mdeim)
  idx_space = get_idx_space(mdeim)

  A = assemble_red_structure(op,μ,meas,field)
  mdeim_online(A,bs,idx_space)
end

function get_parameter(
  op::RBVarOperator{Top,TT,RBSpaceUnsteady},
  μ::Param,
  mdeim::MDEIM,
  info::RBInfo) where {Top,TT}

  bs = get_basis_space(mdeim)
  idx_space = get_idx_space(mdeim)
  timesθ = get_timesθ(op)
  meas = get_reduced_measure(mdeim)

  if info.st_mdeim
    idx_time = get_idx_time(mdeim)
    red_timesθ = timesθ[idx_time]
    A = assemble_red_structure(op,μ,meas,red_timesθ)
    mdeim_online(A,bs,idx_space,get_Nt(op))
  else
    A = assemble_red_structure(op,μ,meas,timesθ)
    interpolate_mdeim_online(A,basis_idx,idx_space,timesθ,idx_time)
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
  assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op),get_test(op))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonlinear,TT,RBSpaceSteady},
  ::Param,
  m::Measure) where TT

  fun = get_fe_function(op)
  M(u) = assemble_matrix(v->fun(m,u,v),get_trial(op),get_test(op))
end

function assemble_red_structure(
  op::RBLinOperator{Nonaffine,RBSpaceUnsteady},
  μ::Param,
  m::Measure,
  timesθ::Vector{Real})

  fun = get_fe_function(op)
  v(tθ) =  assemble_vector(v->fun(μ,tθ,m,v),get_test(op))
  Matrix(v.(timesθ))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonaffine,TT,RBSpaceUnsteady},
  μ::Param,
  m::Measure,
  timesθ::Vector{Real}) where TT

  fun = get_fe_function(op)
  M(tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op),get_test(op))
  Matrix(M.(timesθ))
end

function assemble_red_structure(
  op::RBBilinOperator{Nonlinear,TT,RBSpaceUnsteady},
  ::Param,
  m::Measure,
  timesθ::Vector{Real}) where TT

  fun = get_fe_function(op)
  M(tθ,u) = assemble_matrix(v->fun(μ,tθ,m,u,v),get_trial(op),get_test(op))
  M(u) = Matrix(Broadcasting(tθ -> M(tθ,u))(timesθ))
  M
end

function mdeim_online(
  A::AbstractArray,
  basis_idx::Matrix,
  idx_space::Vector{Int},
  Nt=1)

  param = basis_idx \ reshape(A,:,Nt)[idx_space,:]
  param
end

function mdeim_online(
  A::Function,
  basis_idx::Matrix,
  idx_space::Vector{Int},
  Nt=1)

  param(u) = basis_idx \ reshape(A(u),:,Nt)[idx_space,:]
  param
end

function interpolate_mdeim_online(
  A::AbstractArray,
  basis_idx::Matrix,
  idx_space::Vector{Int},
  timesθ::Vector{Real},
  idx_time::Vector{Int})

  red_timesθ = timesθ[idx_time]
  discarded_idx_time = setdiff(eachindex(timesθ),idx_time)
  red_param = basis_idx \ reshape(A,:,length(idx_time))[idx_space,:]

  itp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:),red_param')
  param = zeros(length(idx_space),length(timesθ))
  param[:,idx_time] = red_param
  for it = discarded_idx_time
    param[:,it] = ScatteredInterpolation.evaluate(itp,[timesθ[it]])
  end

  param
end

function interpolate_mdeim_online(
  A::Function,
  basis_idx::Matrix,
  idx_space::Vector{Int},
  timesθ::Vector{Real},
  idx_time::Vector{Int})

  red_timesθ = timesθ[idx_time]
  discarded_idx_time = setdiff(eachindex(timesθ),idx_time)
  red_param(u) = basis_idx \ reshape(A(u),:,length(idx_time))[idx_space,:]

  itp(u) = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:),red_param(u)')
  param(u)[:,idx_time] = red_param(u)
  for it = discarded_idx_time
    param(u)[:,it] = ScatteredInterpolation.evaluate(itp(u),[timesθ[it]])
  end

  param
end
