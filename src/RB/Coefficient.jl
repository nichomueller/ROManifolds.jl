function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady;
  st_mdeim=false)

  compute_coefficient(op,mdeim,Val(st_mdeim))
end

function compute_coefficient(
  op::RBSteadyVariable{Affine,Ttr},
  args...) where Ttr

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

function setup_coefficient(
  op::RBVariable,
  mdeim::MDEIM,
  args...)

  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space,args...)
  red_lu,A
end

function compute_coefficient(
  op::RBSteadyVariable,
  mdeim::MDEIMSteady)

  red_lu,A = setup_coefficient(op,mdeim)
  (μ::Param) -> mdeim_online(A(μ),red_lu)
end

function compute_coefficient(
  op::RBSteadyVariable{Nonlinear,Ttr},
  mdeim::MDEIMSteady) where Ttr

  red_lu,A = setup_coefficient(op,mdeim)
  (μ::Param,z) -> mdeim_online(A(μ,z),red_lu)
end

function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{false})

  times = get_times(op)
  red_lu,A = setup_coefficient(op,mdeim,times)
  coeff(μ::Param) = mdeim_online(A(μ),red_lu)
  (μ::Param) -> coeff_by_time_bases(op,coeff(μ))
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  mdeim::MDEIMUnsteady,
  ::Val{false}) where Ttr

  times = get_times(op)
  red_lu,A = setup_coefficient(op,mdeim,times)
  coeff(μ::Param,z) = mdeim_online(A(μ,z),red_lu)
  (μ::Param,z) -> coeff_by_time_bases(op,coeff(μ,z))
end

function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{true})

  idx_time = get_idx_time(mdeim)
  red_times = get_times(op)[idx_time]
  red_lu,A = setup_coefficient(op,mdeim,red_times)
  coeff(μ::Param) = mdeim_online(A(μ)[:],red_lu)
  coeff_interp(μ::Param) = interp_coeff_time(mdeim,coeff(μ))
  (μ::Param) -> coeff_by_time_bases(op,coeff_interp(μ))
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  mdeim::MDEIMUnsteady,
  ::Val{true}) where Ttr

  idx_time = get_idx_time(mdeim)
  red_times = get_times(op)[idx_time]
  red_lu,A = setup_coefficient(op,mdeim,red_times)
  coeff(μ::Param,z) = mdeim_online(A(μ,z)[:],red_lu)
  coeff_interp(μ::Param,z) = interp_coeff_time(mdeim,coeff(μ,z))
  (μ::Param,z) -> coeff_by_time_bases(op,coeff_interp(μ,z))
end

function hyperred_structure(
  op::RBSteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_param_fefunction(op)
  test = get_test(op)
  V(μ::Param) = assemble_vector(v->fun(μ,m,v),test)[idx_space]
  V
end

function hyperred_structure(
  op::RBUnsteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{Float})

  fun = get_param_fefunction(op)
  test = get_test(op)
  ns,nt = length(idx_space),length(times)
  V = zeros(ns,nt)

  function V!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      V[:,n] = assemble_vector(v->fun(μ,tn,m,v),test)[idx_space]
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
  test = get_test(op)
  trial = get_trial(op)
  M(μ::Param) = assemble_matrix((u,v)->fun(μ,m,u,v),trial(μ),test)
  μ -> get_findnz_vals(M(μ),idx_space)
end

function hyperred_structure(
  op::RBUnsteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{Float}) where Ttr

  fun = get_param_fefunction(op)
  trial = get_trial(op)
  test = get_test(op)
  ns,nt = length(idx_space),length(times)
  M = zeros(ns,nt)

  function M!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      Mtemp = assemble_matrix((u,v)->fun(μ,tn,m,u,v),trial(μ,tn),test)
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
  test = get_test(op)
  dir(μ::Param) = get_dirichlet_function(op)(μ)
  lift(μ::Param) = assemble_vector(v->fun(μ,m,dir(μ),v),test)[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyLiftVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  times::Vector{Float})

  fun = get_param_fefunction(op)
  test = get_test(op)
  dir(μ::Param,tn::Float) = get_dirichlet_function(op)(μ,tn)
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
  times::Vector{Float}) where Ttr

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
  times::Vector{Float})

  fun = get_param_fefunction(op)
  test = get_test(op)
  dir(μ::Param,tn::Float) = get_dirichlet_function(op)(μ,tn)
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

function mdeim_online(A::Vector{Float},lu::LU)
  P_A = lu.P*A
  y = lu.L \ P_A
  x = lu.U \ y
  x
end

function mdeim_online(A::Matrix{Float},lu::LU)
  P_A = lu.P*A
  y = lu.L \ P_A
  x = lu.U \ y
  Matrix(x')
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
