function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady;
  st_mdeim=false)

  compute_coefficient(op,mdeim,Val(st_mdeim))
end

function compute_coefficient(
  op::RBSteadyVariable{Affine,Ttr},
  args...;kwargs...) where Ttr

  fun = get_param_function(op)
  coeff(μ::Param) = [fun(nothing,μ)[1]]

  coeff
end

function compute_coefficient(
  op::RBUnsteadyVariable{Affine,Ttr},
  args...;kwargs...) where Ttr

  fun = get_param_function(op)
  times = get_times(op)
  ns,nt = 1,length(times)
  coeff = zeros(ns,nt)

  function coeff!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      coeff[:,n] = fun(nothing,μ,tn)[1]
    end
    coeff
  end

  coeff_bt(μ::Param) = coeff_by_time_bases(op,coeff(μ))
  coeff_bt
end

function compute_coefficient(
  op::RBSteadyVariable,
  mdeim::MDEIMSteady;
  kwargs...)

  nnl = isnonlinear(op)

  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space)
  coeff = mdeim_online(A,red_lu,Val(nnl))

  coeff
end

function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{false})

  nnl = isnonlinear(op)

  times = get_times(op)
  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space,times)
  coeff = mdeim_online(A,red_lu,Val(nnl))
  coeff_bt = coeff_by_time_bases(op,coeff)

  coeff_bt
end

function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{true})

  spacetime_vector(fun::Function,::Val{false}) = μ -> fun(μ)[:]
  spacetime_vector(fun::Function,::Val{true}) = (μ,z) -> fun(μ,z)[:]

  nnl = isnonlinear(op)

  red_lu = get_red_lu_factors(mdeim)
  idx_space,idx_time = get_idx_space(mdeim),get_idx_time(mdeim)
  red_times = get_times(op)[idx_time]
  m = get_red_measure(mdeim)

  A_idx = hyperred_structure(op,m,idx_space,red_times)
  A_st = spacetime_vector(A_idx,Val(nnl))
  coeff = mdeim_online(A_st,red_lu,Val(nnl))
  coeff_interp = interp_coeff_time(mdeim,coeff,Val(nnl))
  coeff_bt = coeff_by_time_bases(op,coeff_interp)

  coeff_bt
end

function hyperred_structure(
  op::RBSteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_param_fefunction(op)
  V(μ::Param) = assemble_vector(v->fun(μ,m,v),get_test(op))[idx_space]
  V
end

function hyperred_structure(
  op::RBUnsteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{<:Real})

  fun = get_param_fefunction(op)
  test = get_test(op)
  ns,nt = length(idx_space),length(times)
  V = zeros(ns,nt)

  function V!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      V[:,n] = assemble_vector(v->fun(μ,tn,m,v),test(op))[idx_space]
    end
    V
  end

  V!
end

function hyperred_structure(
  op::RBSteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int}) where Ttr

  fun = get_param_fefunction(op)
  M(μ::Param) = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
  μ -> get_findnz_vals(M(μ),idx_space)
end

function hyperred_structure(
  op::RBUnsteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{<:Real}) where Ttr

  fun = get_param_fefunction(op)
  trial = get_trial(op)
  test = get_test(op)
  ns,nt = length(idx_space),length(times)
  M = zeros(ns,nt)

  function M!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      Mtemp = assemble_matrix((u,v)->fun(μ,tn,m,u,v),trial(op)(μ,tn),test(op))
      M[:,n] = get_findnz_vals(Mtemp,idx_space)
    end
    M
  end

  M!
end

function hyperred_structure(
  op::RBSteadyLiftVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_param_fefunction(op)
  dir(μ::Param) = get_dirichlet_function(op)(μ)
  lift(μ::Param) = assemble_vector(v->fun(μ,m,dir(μ),v),get_test(op))[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyLiftVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{<:Real})

  fun = get_param_fefunction(op)
  test = get_test(op)
  dir(μ::Param,tn::Real) = get_dirichlet_function(op)(μ,tn)
  ns,nt = length(idx_space),length(times)
  lift = zeros(ns,nt)

  function lift!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      lift[:,n] = assemble_vector(v->fun(μ,tn,m,dir(μ,tn),v),test)[idx_space]
    end
    lift
  end

  lift!
end

function hyperred_structure(
  op::RBSteadyBilinVariable{Nonlinear,Ttr},
  m::Measure,
  idx_space::Vector{Int}) where Ttr

  fun = get_param_fefunction(op)
  trial = get_trial(op)
  test = get_test(op)

  M(μ::Param,z) = assemble_matrix((u,v)->fun(m,z,u,v),trial(μ),test)
  (μ::Param,z) -> get_findnz_vals(M(μ,z),idx_space)
end

function hyperred_structure(
  op::RBUnsteadyBilinVariable{Nonlinear,Ttr},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{<:Real}) where Ttr

  fun = get_param_fefunction(op)
  trial = get_trial(op)
  test = get_test(op)
  ns,nt = length(idx_space),length(times)
  M = zeros(ns,nt)

  function M!(μ::Param,z)
    @inbounds for (n,tn) in enumerate(times)
      Mtemp = assemble_matrix((u,v)->fun(m,z(tn),u,v),trial(μ,tn),test)
      M[:,n] = get_findnz_vals(Mtemp,idx_space)
    end
    M
  end

  M!
end

function hyperred_structure(
  op::RBSteadyLiftVariable{Nonlinear},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_param_fefunction(op)
  test = get_test(op)
  dir(μ::Param) = get_dirichlet_function(op)(μ)
  lift(μ::Param) = assemble_vector(v->fun(m,z,dir(μ),v),test)[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyLiftVariable{Nonlinear},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{<:Real})

  fun = get_param_fefunction(op)
  test = get_test(op)
  dir(μ::Param,tn::Real) = get_dirichlet_function(op)(μ,tn)
  ns,nt = length(idx_space),length(times)
  lift = zeros(ns,nt)

  function lift!(μ::Param,z)
    @inbounds for (n,tn) in enumerate(times)
      lift[:,n] = assemble_vector(v->fun(m,z(tn),dir(μ,tn),v),test)[idx_space]
    end
    lift
  end

  lift!
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
  red_lu::LU,
  ::Val{false})

  (μ::Param) -> mdeim_online(A(μ),red_lu)
end

function mdeim_online(
  A::Function,
  red_lu::LU,
  ::Val{true})

  (μ::Param,z) -> mdeim_online(A(μ,z),red_lu)
end

function interp_coeff_time(
  mdeim::MDEIMUnsteady,
  coeff::Vector{Float})

  bs = get_basis_space(mdeim)
  bt = get_basis_time(mdeim)
  Qs = size(bs,2)
  Qt = size(bt,2)
  sorted_idx(qs) = [(i-1)*Qs+qs for i = 1:Qt]

  interp_coeff = zeros(Qt,Qs)
  @inbounds for qs = 1:Qs
    interp_coeff[:,qs] = bt*coeff[sorted_idx(qs)]
  end

  interp_coeff
end

function interp_coeff_time(
  mdeim::MDEIMUnsteady,
  coeff::Function,
  ::Val{false})

  (μ::Param) -> interp_coeff_time(mdeim,coeff(μ))
end

function interp_coeff_time(
  mdeim::MDEIMUnsteady,
  coeff::Function,
  ::Val{true})

  (μ::Param,z) -> interp_coeff_time(mdeim,coeff(μ,z))
end

function coeff_by_time_bases(
  op::RBUnsteadyVariable,
  coeff::AbstractMatrix)

  rbrow = get_rbspace_row(op)
  rb_time_projection(rbrow,coeff)
end

function coeff_by_time_bases(
  op::RBUnsteadyBilinVariable,
  coeff::AbstractMatrix)

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  time_proj(idx1,idx2) = rb_time_projection(rbrow,rbcol,coeff;
    idx_forwards=idx1,idx_backwards=idx2)

  θ = get_θ(op)
  dt = get_dt(op)
  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt

  btbtc = time_proj(idx,idx)
  btbtc_shift = time_proj(idx_forwards,idx_backwards)

  if get_id(op) == :M
    btbtc/dt - btbtc_shift/dt
  else
    θ*btbtc + (1-θ)*btbtc_shift
  end
end

function coeff_by_time_bases(
  op::RBUnsteadyVariable,
  coeff::Function)

  (μ::Param) -> coeff_by_time_bases(op,coeff(μ))
end

function coeff_by_time_bases(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  coeff::Function) where Ttr

  (μ::Param,z) -> coeff_by_time_bases(op,coeff(μ,z))
end
