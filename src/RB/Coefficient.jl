function compute_coefficient(
  op::RBSteadyVariable{Affine,Ttr},
  args...;kwargs...) where Ttr

  fun = get_param_function(op)
  coeff(μ) = [fun(nothing,μ)[1]]

  coeff
end

function compute_coefficient(
  op::RBUnsteadyVariable{Affine,Ttr},
  args...;kwargs...) where Ttr

  fun = get_param_function(op)
  timesθ = get_timesθ(op)
  coeff(μ,tθ) = fun(nothing,μ,tθ)[1]
  coeff(μ) = Matrix(Broadcasting(tθ -> coeff(μ,tθ))(timesθ))
  coeff_bt(μ) = coeff_by_time_bases(op,coeff(μ))

  coeff_bt
end

function compute_coefficient(
  op::RBSteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMSteady;
  kwargs...) where Ttr

  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space)
  coeff(μ) = mdeim_online(A(μ),red_lu)

  coeff
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMUnsteady;
  st_mdeim=false) where Ttr

  compute_coefficient(op,mdeim,Val(st_mdeim))
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMUnsteady,
  ::Val{false}) where Ttr

  timesθ = get_timesθ(op)
  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space,timesθ)
  coeff(μ) = mdeim_online(A(μ),red_lu)
  coeff_bt(μ) = coeff_by_time_bases(op,coeff(μ))

  coeff_bt
end

function compute_coefficient(
  op::RBUnsteadyVariable{Nonaffine,Ttr},
  mdeim::MDEIMUnsteady,
  ::Val{true}) where Ttr

  red_lu = get_red_lu_factors(mdeim)
  idx_space,idx_time = get_idx_space(mdeim),get_idx_time(mdeim)
  red_timesθ = get_reduced_timesθ(op,idx_time)
  m = get_red_measure(mdeim)

  A_idx = hyperred_structure(op,m,idx_space,red_timesθ)
  A_st(μ) = spacetime_vector(A_idx(μ))
  coeff(μ) = mdeim_online(A_st(μ),red_lu)
  coeff_interp(μ) = interp_coeff_time(mdeim,coeff(μ))
  coeff_bt(μ) = coeff_by_time_bases(op,coeff_interp(μ))

  coeff_bt
end

function compute_coefficient(
  ::RBVariable{Nonlinear,Ttr},
  args...;
  kwargs...) where Ttr

  coeff(u::Vector) = u#[Matrix([u[i]]) for i=eachindex(u)]
  coeff
end

function hyperred_structure(
  op::RBSteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  V(μ) = assemble_vector(v->fun(μ,m,v),get_test(op))[idx_space]
  V
end

function hyperred_structure(
  op::RBUnsteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  V(μ,tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))[idx_space]
  V(μ) = Matrix(Broadcasting(tθ -> V(μ,tθ))(timesθ))
  V
end

function hyperred_structure(
  op::RBSteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int}) where Ttr

  fun = get_fe_function(op)
  M(μ) = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
  μ -> Vector(M(μ)[:][idx_space])
end

function hyperred_structure(
  op::RBUnsteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real}) where Ttr

  fun = get_fe_function(op)
  M(μ,tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(μ,tθ) = Vector(M(μ,tθ)[:][idx_space])
  Midx(μ) = Matrix(Broadcasting(tθ -> Midx(μ,tθ))(timesθ))
  Midx
end

function hyperred_structure(
  op::RBSteadyLiftVariable,
  m::Measure,
  idx_space::Vector{Int})

  fun = get_fe_function(op)
  dir(μ) = get_dirichlet_function(op)(μ)
  fdofs_test = get_fdofs_on_full_trian(get_tests(op))
  ddofs = get_ddofs_on_full_trian(get_trials(op))

  Mlift(μ) = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(μ) = (Mlift(μ)[fdofs_test,ddofs]*dir(μ))[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyVariable,
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_fe_function(op)
  dir(μ,tθ) = get_dirichlet_function(op)(μ,tθ)
  fdofs_test = get_fdofs_on_full_trian(get_tests(op))
  ddofs = get_ddofs_on_full_trian(get_trials(op))

  Mlift(μ,tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial_no_bc(op),get_test_no_bc(op))
  lift(μ,tθ) = (Mlift(μ,tθ)[fdofs_test,ddofs]*dir(μ,tθ))[idx_space]
  lift(μ) = Matrix(Broadcasting(tθ -> lift(μ,tθ))(timesθ))

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

function coeff_by_time_bases(
  op::RBVariable,
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

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt

  btbtc = time_proj(idx,idx)
  btbtc_shift = time_proj(idx_forwards,idx_backwards)
  btbtc,btbtc_shift
end
