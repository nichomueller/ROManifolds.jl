function get_coefficient(
  op::RBSteadyVariable{Affine,Ttr},
  args...) where Ttr

  fun = get_param_function(op)
  (μ::Param) -> fun(nothing,μ)[1]
end

function get_coefficient(
  op::RBSteadyVariable,
  mdeim::MDEIMSteady)

  mdeim_solver,A = setup_coefficient(op,mdeim)
  (μ::Param) -> mdeim_solver(A(μ))
end

function get_coefficient(
  op::RBSteadyVariable{Nonlinear,Ttr},
  mdeim::MDEIMSteady) where Ttr

  mdeim_solver,A = setup_coefficient(op,mdeim)
  (μ::Param,z) -> mdeim_solver(A(μ,z))
end

function get_coefficient(
  op::RBUnsteadyVariable{Affine,Ttr},
  basis::AbstractMatrix;
  kwargs...) where Ttr

  fun = get_param_function(op)
  times = get_times(op)
  Nt,Qs = length(times),1
  coeff = allocate_matrix(Matrix{Float},Nt,Qs)

  function coeff!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      coeff[n,:] .= fun(nothing,μ,tn)[1]
    end
    coeff
  end

  @assert Qs == size(basis,2) "Wrong dimensions"
  coeff_by_time_bases = get_coeff_by_time_bases(op,Qs)
  (μ::Param) -> coeff_by_time_bases(coeff!(μ))
end

function get_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady;
  st_mdeim=false)

  get_coefficient(op,mdeim,Val(st_mdeim))
end

function get_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{false})

  times = get_times(op)
  mdeim_solver,A = setup_coefficient(op,mdeim,times)
  coeff_by_time_bases = get_coeff_by_time_bases(op,mdeim)

  coeff(μ::Param) = mdeim_solver(A(μ))
  (μ::Param) -> coeff_by_time_bases(coeff(μ))
end

function get_coefficient(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  mdeim::MDEIMUnsteady,
  ::Val{false}) where Ttr

  times = get_times(op)
  mdeim_solver,A = setup_coefficient(op,mdeim,times)
  coeff_by_time_bases = get_coeff_by_time_bases(op,mdeim)

  coeff(μ::Param,z) = mdeim_solver(A(μ,z))
  (μ::Param,z) -> coeff_by_time_bases(coeff(μ,z))
end

function get_coefficient(
  op::RBUnsteadyVariable,
  mdeim::MDEIMUnsteady,
  ::Val{true})

  idx_time = get_idx_time(mdeim)
  red_times = get_times(op)[idx_time]
  mdeim_solver,A = setup_coefficient(op,mdeim,red_times)
  coeff_by_time_bases = get_coeff_by_time_bases(op,mdeim)
  interp_coeff = get_interp_coeff(mdeim)

  coeff(μ::Param) = mdeim_solver(reshape(A(μ),:,1))
  (μ::Param) -> coeff_by_time_bases(interp_coeff(coeff(μ)))
end

function get_coefficient(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  mdeim::MDEIMUnsteady,
  ::Val{true}) where Ttr

  idx_time = get_idx_time(mdeim)
  red_times = get_times(op)[idx_time]
  mdeim_solver,A = setup_coefficient(op,mdeim,red_times)
  coeff_by_time_bases = get_coeff_by_time_bases(op,mdeim)
  interp_coeff = get_interp_coeff(mdeim)

  coeff(μ::Param,z) = mdeim_solver(A(μ,z)[:])
  (μ::Param,z) -> coeff_by_time_bases(interp_coeff(coeff(μ,z)))
end

function setup_coefficient(
  op::RBVariable,
  mdeim::MDEIM,
  args...)

  red_lu = get_rb_lu(mdeim)
  idx_space = get_idx_space(mdeim)
  m = get_red_measure(mdeim)
  A = hyperred_structure(op,m,idx_space,args...)
  mdeim_solver = mdeim_online(red_lu)
  mdeim_solver,A
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
  V = allocate_matrix(Matrix{Float},ns,nt)

  function V!(μ::Param)
    @inbounds for (n,tn) in enumerate(times)
      copyto!(view(V,:,n),assemble_vector(v->fun(μ,tn,m,v),test)[idx_space])
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
  M = allocate_matrix(Matrix{Float},ns,nt)

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
  lift = allocate_matrix(Matrix{Float},ns,nt)

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
  M = allocate_matrix(Matrix{Float},ns,nt)

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
  lift = allocate_matrix(Matrix{Float},ns,nt)

  function lift!(μ::Param,z)
    @inbounds for (n,tn) in enumerate(times)
      lift[:,n] = assemble_vector(v->fun(m,z(tn),dir(μ,tn),v),test)[idx_space]
    end
    lift
  end

  lift!
end

function mdeim_online(lu_rb::LU)

  function sol!(A::AbstractMatrix{Float})
    P_A = lu_rb.P*A
    y = lu_rb.L \ P_A
    x = lu_rb.U \ y
    x'
  end

  sol!
end

function get_interp_coeff(mdeim::MDEIMUnsteady)
  bs = get_basis_space(mdeim)
  bt = get_basis_time(mdeim)
  Qs = size(bs,2)
  Nt,Qt = size(bt)
  sorted_idx(qs) = [(i-1)*Qs+qs for i = 1:Qt]

  interp_coeff = allocate_matrix(Matrix{Float},Nt,Qs)
  function interp_coeff!(coeff::AbstractMatrix)
    @inbounds for qs = 1:Qs
      interp_coeff[:,qs] = bt*coeff[sorted_idx(qs)]
    end
    interp_coeff
  end

  interp_coeff!
end

function get_coeff_by_time_bases(op::RBUnsteadyVariable,mdeim::MDEIMUnsteady)
  Qs = size(get_basis_space(mdeim),2)
  get_coeff_by_time_bases(op,Qs)
end

function get_coeff_by_time_bases(op::RBUnsteadyVariable,Qs::Int)
  rbrow = get_rbspace_row(op)
  brow = get_basis_time(rbrow)
  nrow = size(brow,2)

  proj = allocate_matrix(Matrix{Float},nrow,Qs)
  function time_proj!(coeff::AbstractMatrix)
    @assert size(coeff,2) == Qs "Dimension mismatch: $(size(coeff,2)) != $Qs"

    @inbounds for q = 1:Qs, it = 1:nrow
      proj[it,q] = sum(brow[:,it].*coeff[:,q])
    end
    proj
  end

  time_proj!
end

function get_coeff_by_time_bases(op::RBUnsteadyBilinVariable,Qs::Int)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  brow = get_basis_time(rbrow)
  bcol = get_basis_time(rbcol)
  nrow = size(brow,2)
  ncol = size(bcol,2)
  Nt = get_Nt(op)
  θ = get_θ(op)
  dt = get_dt(op)

  proj = allocate_matrix(Matrix{Float},nrow*ncol,Qs)
  proj_shift = allocate_matrix(Matrix{Float},nrow*ncol,Qs)
  function time_proj!(coeff::AbstractMatrix)
    @assert size(coeff,2) == Qs "Dimension mismatch: $(size(coeff,2)) != $Qs"
    @inbounds for q = 1:Qs, jt = 1:ncol, it = 1:nrow
      proj[(jt-1)*nrow+it,q] = sum(brow[:,it].*bcol[:,jt].*coeff[:,q])
      proj_shift[(jt-1)*nrow+it,q] = sum(brow[2:Nt,it].*bcol[1:Nt-1,jt].*coeff[2:Nt,q])
    end

    if get_id(op) == :M
      return proj/dt - proj_shift/dt
    else
      return θ*proj + (1-θ)*proj_shift
    end
  end

  time_proj!
end

function get_coeff_by_time_bases_try(op::RBUnsteadyBilinVariable,Qs::Int)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  brow = get_basis_time(rbrow)
  bcol = get_basis_time(rbcol)
  nrow = size(brow,2)
  ncol = size(bcol,2)
  Nt = get_Nt(op)
  θ = get_θ(op)
  dt = get_dt(op)

  btbt = allocate_matrix(Matrix{Float},Nt,nrow*ncol)
  btbt_shift = allocate_matrix(Matrix{Float},Nt-1,nrow*ncol)
  @inbounds for jt = 1:ncol, it = 1:nrow
    btbt[:,(jt-1)*nrow+it] .= brow[:,it].*bcol[:,jt]
    btbt_shift[:,(jt-1)*nrow+it] .= brow[2:Nt,it].*bcol[1:Nt-1,jt]
  end

  proj = allocate_matrix(Matrix{Float},nrow*ncol,Qs)
  proj_shift = allocate_matrix(Matrix{Float},nrow*ncol,Qs)
  function time_proj!(coeff::AbstractMatrix)
    @assert size(coeff,2) == Qs "Dimension mismatch: $(size(coeff,2)) != $Qs"
    @inbounds for q = 1:Qs, ijt = 1:ncol*nrow
      proj[ijt,q] = sum(btbt[:,ijt].*coeff[:,q])
      proj_shift[ijt,q] = sum(btbt_shift[:,ijt].*coeff[2:Nt,q])
    end

    if get_id(op) == :M
      return proj/dt - proj_shift/dt
    else
      return θ*proj + (1-θ)*proj_shift
    end
  end

  time_proj!
end
