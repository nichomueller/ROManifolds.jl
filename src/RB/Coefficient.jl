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

  timesθ = get_timesθ(op)
  red_lu = get_red_lu_factors(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space,timesθ)
  coeff = mdeim_online(A,red_lu,Val(nnl))
  coeff_bt = coeff_by_time_bases(op,coeff)

  coeff_bt
end

function compute_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{true})

  spacetime_vector(fun::Function,::Val{false}) = μ -> fun(μ)[:]
  spacetime_vector(fun::Function,::Val{true}) = (μ,u) -> fun(μ,u)[:]

  nnl = isnonlinear(op)

  red_lu = get_red_lu_factors(mdeim)
  idx_space,idx_time = get_idx_space(mdeim),get_idx_time(mdeim)
  red_timesθ = get_timesθ(op)[idx_time]
  m = get_red_measure(mdeim)

  A_idx = hyperred_structure(op,m,idx_space,red_timesθ)
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
  V(μ) = assemble_vector(v->fun(μ,m,v),get_test(op))[idx_space]
  V
end

function hyperred_structure(
  op::RBUnsteadyLinVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_param_fefunction(op)
  V(μ,tθ) = assemble_vector(v->fun(μ,tθ,m,v),get_test(op))[idx_space]
  V(μ) = Matrix(Broadcasting(tθ -> V(μ,tθ))(timesθ))
  V
end

function hyperred_structure(
  op::RBSteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int}) where Ttr

  fun = get_param_fefunction(op)
  M(μ) = assemble_matrix((u,v)->fun(μ,m,u,v),get_trial(op)(μ),get_test(op))
  μ -> Vector(M(μ)[:][idx_space])
end

function hyperred_structure(
  op::RBUnsteadyBilinVariable{Nonaffine,Ttr},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real}) where Ttr

  fun = get_param_fefunction(op)
  M(μ,tθ) = assemble_matrix((u,v)->fun(μ,tθ,m,u,v),get_trial(op)(μ,tθ),get_test(op))
  Midx(μ,tθ) = Vector(M(μ,tθ)[:][idx_space])
  Midx(μ) = Matrix(Broadcasting(tθ -> Midx(μ,tθ))(timesθ))
  Midx
end

function hyperred_structure(
  op::RBSteadyLiftVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_param_fefunction(op)
  dir(μ) = get_dirichlet_function(op)(μ)
  lift(μ) = assemble_vector(v->fun(μ,m,dir(μ),v),get_test(op))[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyLiftVariable{Nonaffine},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_param_fefunction(op)
  dir(μ,tθ) = get_dirichlet_function(op)(μ,tθ)
  lift(μ,tθ) = assemble_vector(v->fun(μ,tθ,m,dir(μ,tθ),v),get_test(op))[idx_space]
  lift(μ) = Matrix(Broadcasting(tθ -> lift(μ,tθ))(timesθ))

  lift
end

function hyperred_structure(
  op::RBSteadyBilinVariable{Nonlinear,Ttr},
  m::Measure,
  idx_space::Vector{Int}) where Ttr

  fun = get_param_fefunction(op)
  trial = get_trial(op)
  test = get_test(op)

  M(μ::Param,z) = assemble_matrix((u,v)->fun(m,z,u,v),trial(μ),test)
  (μ::Param,z) -> Vector(M(μ,z)[:][idx_space])
end

function hyperred_structure(
  op::RBUnsteadyBilinVariable{Nonlinear,Ttr},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real}) where Ttr

  fun = get_param_fefunction(op)
  trial = get_trial(op)
  test = get_test(op)

  M(μ::Param,tθ::Real,z) = assemble_matrix((u,v)->fun(m,z(tθ),u,v),trial(μ,tθ),test)
  Midx(μ::Param,tθ::Real,z) = Vector(M(μ,tθ,z)[:][idx_space])
  Midx(μ::Param,z) = Matrix(Broadcasting(tθ -> Midx(μ,tθ,z))(timesθ))
  Midx
end

function hyperred_structure(
  op::RBSteadyLiftVariable{Nonlinear},
  m::Measure,
  idx_space::Vector{Int})

  fun = get_param_fefunction(op)
  dir(μ) = get_dirichlet_function(op)(μ)
  lift(μ) = assemble_vector(v->fun(m,z,dir(μ),v),get_test(op))[idx_space]

  lift
end

function hyperred_structure(
  op::RBUnsteadyLiftVariable{Nonlinear},
  m::Measure,
  idx_space::Vector{Int},
  timesθ::Vector{<:Real})

  fun = get_param_fefunction(op)
  dir(μ,tθ) = get_dirichlet_function(op)(μ,tθ)
  lift(μ,tθ,z) = assemble_vector(v->fun(m,z(tθ),dir(μ,tθ),v),get_test(op))[idx_space]
  lift(μ,z) = Matrix(Broadcasting(tθ -> lift(μ,tθ,z))(timesθ))

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

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt

  btbtc = time_proj(idx,idx)
  btbtc_shift = time_proj(idx_forwards,idx_backwards)
  btbtc,btbtc_shift
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
