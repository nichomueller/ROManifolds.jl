function compute_coefficient(op::RBSteadyVariable{Affine,Ttr}) where Ttr
  fun = get_param_function(op)
  μ = realization(op)
  coeff = first(fun(nothing,μ))
  [coeff]
end

function compute_coefficient(op::RBUnsteadyVariable{Affine,Ttr}) where Ttr
  fun = get_param_function(op)
  μ = realization(op)
  timesθ = get_timesθ(op)
  coeff(tθ) = first(fun(nothing,μ,tθ))
  Matrix(coeff.(timesθ))
end

function compute_coefficient(
  op::RBSteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMSteady,
  μ::Param) where Ttr

  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)

  A = hyperred_structure(op,m,μ,idx_space)
  mdeim_online(A,red_lu)
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMUnsteady,
  μ::Param,
  st_mdeim=false) where Ttr

  compute_coefficient(op,mdeim,μ,Val(st_mdeim))
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMUnsteady,
  μ::Param,
  ::Val{false}) where Ttr

  timesθ = get_timesθ(op)
  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,μ,idx_space,timesθ)

  mdeim_online(A,red_lu)
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMUnsteady,
  μ::Param,
  ::Val{true}) where Ttr

  red_lu = get_red_lu_factors(mdeim)
  idx_space,idx_time = get_idx_space(mdeim),get_idx_time(mdeim)
  red_timesθ = get_reduced_timesθ(op,idx_time)
  m = get_red_measure(mdeim)

  A_idx = hyperred_structure(op,m,μ,idx_space,red_timesθ)
  A_st = spacetime_vector(A_idx)
  coeff = mdeim_online(A_st,red_lu)
  interp_coeff_time(mdeim,coeff)
end

function compute_coefficient(::RBSteadyVariable{Nonlinear,Ttr}) where Ttr
  coeff(u) = u
  coeff
end

function compute_coefficient(::RBUnsteadyVariable{Nonlinear,Ttr}) where Ttr
  coeff(u,tθ) = u(tθ)
  coeff(tθ) = u -> u(tθ)
  coeff
end

function hyperred_structure(
  op::RBSteadyLinVariable{Nonaffine},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  assemble_vector(v->fun(μ,m,v),get_test(op))[idx_space]
end

function hyperred_structure(
  op::RBUnsteadyLinVariable{Nonaffine},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  v(tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))[idx_space]
  Matrix(v.(timesθ))
end

function hyperred_structure(
  op::RBSteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  μ::Param,
  idx_space::Vector{Int}) where Ttr

  fun = get_fe_function(op)
  M = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
  Vector(M[:][idx_space])
end

function hyperred_structure(
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

function hyperred_structure(
  op::RBSteadyLiftVariable,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  dir = get_dirichlet_function(op)(μ)
  fdofs_test = get_fdofs_on_full_trian(get_tests(op))
  ddofs = get_ddofs_on_full_trian(get_trials(op))

  Mlift = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift = (Mlift[fdofs_test,ddofs]*dir)[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyVariable,
  m::Measure,
  μ::Param,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  dir(tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs_test = get_fdofs_on_full_trian(get_tests(op))
  ddofs = get_ddofs_on_full_trian(get_trials(op))

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
  red_lu::LU)

  solve_lu(Aidx,red_lu)
end

function mdeim_online(
  A::Function,
  red_lu::LU)

  u -> mdeim_online(A(u),red_lu)
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

function rb_online_structure(
  basis::Matrix{Float},
  coeff::Vector{Float},
  nr::Int)

  @assert size(basis,2) == length(coeff) "Something is wrong"
  bc = sum([basis[:,k]*coeff[k] for k=eachindex(coeff)])
  Matrix(reshape(bc,nr,:))
end

function rb_online_structure(
  basis::BlockMatrix{Float},
  coeff::BlockMatrix{Float},
  nr::Int)

  @assert length(basis) == length(coeff) "Something is wrong"
  bc = sum([kron(basis[k],coeff[k]) for k=eachindex(coeff)])
  Matrix(reshape(bc,nr,:))
end

function rb_online_structure(
  basis::BlockMatrix{Float},
  coeff::NTuple{2,BlockMatrix{Float}},
  nr::Int)

  Broadcasting(c->rb_online_structure(basis,c,nr))(coeff)
end
