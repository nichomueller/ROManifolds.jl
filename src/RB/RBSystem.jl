function online_assembler(
  op::RBVarOperator,
  basis::NTuple{2,Matrix{Float}},
  μ::Param)

  Broadcasting(b->online_assembler(op,b,μ))(basis)
end

function online_assembler(
  op::RBVarOperator,
  basis::Matrix{Float},
  μ::Param)

  coeff = compute_coefficient(op,μ)
  online_structure(op,basis,coeff)
end

function online_assembler(
  op::RBVarOperator,
  mdeim::Union{MDEIM,NTuple{2,MDEIM}},
  μ::Param,
  args...)

  basis = get_basis_space(mdeim)
  coeff = compute_coefficient(op,mdeim,μ,args...)
  online_structure(op,basis,coeff)
end

function online_assembler(
  op::RBBilinOperator,
  basis_mdeim::Tuple{Matrix{Float},MDEIM},
  μ::Param,
  args...)

  affine_basis,mdeim = basis_mdeim
  affine_structure = online_assembler(op,affine_basis,μ)

  op_lift = RBLiftingOperator(op)
  nonaffine_basis = get_basis_space(mdeim)
  coeff = compute_coefficient(op_lift,mdeim,μ,args...)
  nonaffine_structure = online_structure(op_lift,nonaffine_basis,coeff)

  affine_structure,nonaffine_structure
end

function online_structure(
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  basis::Union{Matrix{Float},NTuple{2,Matrix{Float}}},
  coeff::Any)

  nr = get_nrows(op)
  basis_by_coeff_mult(basis,coeff,nr)
end

function online_structure(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  basis::Union{Matrix{Float},NTuple{2,Matrix{Float}}},
  coeff::Any)

  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff /= dtθ end

  btbtp = coeff_by_time_bases(op,coeff)
  ns_row = get_ns(get_rbspace_row(op))

  basis_block = blocks(basis,ns_row)

  nr = get_nrows(op)
  basis_by_coeff_mult(basis_block,btbtp,nr)
end

function online_structure(
  op::Union{RBSteadyLinOperator{Nonlinear},RBSteadyBilinOperator{Nonlinear,TT},RBSteadyLiftingOperator{Nonlinear,TT}},
  basis::Union{Matrix{Float},NTuple{2,Matrix{Float}}},
  coeff::Any) where TT

  nr = get_nrows(op)
  u -> basis_by_coeff_mult(basis,coeff(u),nr)
end

function online_structure(
  op::Union{RBUnsteadyLinOperator{Nonlinear},RBUnsteadyBilinOperator{Nonlinear,TT},RBUnsteadyLiftingOperator{Nonlinear,TT}},
  basis::Union{Matrix{Float},NTuple{2,Matrix{Float}}},
  coeff::Any) where TT

  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff /= dtθ end

  btbtp = coeff_by_time_bases(op,coeff)
  nr = get_nrows(op)
  nc = get_ncols(op)
  basis_block = blocks(basis,size(basis,2);dims=(nr,nc))

  u -> basis_by_coeff_mult(basis_block,btbtp(u),nr)
end

function coeff_by_time_bases(op::RBUnsteadyLinOperator,coeff)
  coeff_by_time_bases_lin(op,coeff)
end

function coeff_by_time_bases(op::RBUnsteadyBilinOperator,coeff)
  coeff_by_time_bases_bilin(op,coeff)
end

function coeff_by_time_bases(op::RBUnsteadyLiftingOperator,coeff)
  coeff_by_time_bases_lin(op,coeff)
end

function coeff_by_time_bases(
  op::RBUnsteadyBilinOperator,
  coeff::NTuple{2,T}) where T

  coeff_by_time_bases_bilin(op,first(coeff)),coeff_by_time_bases_lin(op,last(coeff))
end

function coeff_by_time_bases_lin(
  op::RBVarOperator,
  coeff::AbstractMatrix)

  bt_row = get_basis_time_row(op)
  nt_row = size(bt_row,2)
  Q = size(coeff,1)

  btp_fun(it,q) = sum(bt_row[:,it].*coeff[q,:])
  btp_fun(q) = Matrix(Broadcasting(it -> btp_fun(it,q))(1:nt_row))
  btp_fun.(1:Q)
end

function coeff_by_time_bases_lin(
  op::RBVarOperator{Nonlinear,ParamTransientTrialFESpace},
  coeff::Function)

  bt_row = get_basis_time_row(op)
  nt_row = size(bt_row,2)
  Q = size(coeff,1)

  btp_fun(u,it,q) = sum(bt_row[:,it].*coeff(u)[q,:])
  btp_fun(u,q) = Matrix(Broadcasting(it -> btp_fun(u,it,q))(1:nt_row))
  btp_fun(u) = Broadcasting(q -> btp_fun(u,q))(1:Q)
  btp_fun
end

function coeff_by_time_bases_bilin(
  op::RBVarOperator,
  coeff::AbstractMatrix)

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_shift_backward,idx_shift_forward = 1:Nt-1,2:Nt
  bt_row = get_basis_time_row(op)
  bt_col = get_basis_time_col(op)

  nt_row = size(bt_row,2)
  nt_col = size(bt_col,2)
  Q = size(coeff,1)

  function define_btbtp_fun(idx1,idx2)
    btbtp_fun(it,jt,q) = sum(bt_row[idx1,it].*bt_col[idx2,jt].*coeff[q,idx1])
    btbtp_fun(jt,q) = Broadcasting(it -> btbtp_fun(it,jt,q))(1:nt_row)
    btbtp_fun(q) = Broadcasting(jt -> btbtp_fun(jt,q))(1:nt_col)
    btbtp_block = Matrix.(btbtp_fun.(1:Q))
    btbtp_block
  end

  M = define_btbtp_fun(idx,idx)
  M_shift = define_btbtp_fun(idx_shift_forward,idx_shift_backward)
  M,M_shift
end

function coeff_by_time_bases_bilin(
  op::RBVarOperator{Nonlinear,ParamTransientTrialFESpace},
  coeff::Function)

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_shift_back,idx_shift_forward = 1:Nt-1,2:Nt
  bt_row = get_basis_time_row(op)
  bt_col = get_basis_time_col(op)

  nt_row = size(bt_row)[2]
  nt_col = size(bt_col)[2]
  Q = size(coeff,1)

  function define_btbtp_fun(idx1,idx2)
    btbtp_fun(u,it,jt,q) = sum(bt_row[idx1,it].*bt_col[idx2,jt].*coeff(u)[q,idx1])
    btbtp_fun(u,jt,q) = Broadcasting(it -> btbtp_fun(u,it,jt,q))(1:nt_row)
    btbtp_fun(u,q) = Broadcasting(jt -> btbtp_fun(u,jt,q))(1:nt_col)
    btbtp_fun(u) = Matrix.(Broadcasting(q -> btbtp_fun(u,q))(1:Q))
    btbtp_fun
  end

  M = define_btbtp_fun(idx,idx)
  M_shift = define_btbtp_fun(idx_shift_forward,idx_shift_back)
  M,M_shift
end

function poisson_rb_system(
  lhs::Matrix{Float},
  rhs::NTuple{N,Matrix{Float}}) where N

  A_rb = lhs
  F_rb,H_rb,lifts... = rhs
  A_rb,F_rb+H_rb-sum(lifts)
end

function poisson_rb_system(
  lhs::NTuple{4,Matrix{Float}},
  rhs::NTuple{N,Matrix{Float}},
  θ::Float) where N

  A_rb,Ashift_rb,M_rb,Mshift_rb = lhs
  F_rb,H_rb,lifts... = rhs

  rb_lhs = θ*(A_rb+M_rb) + (1-θ)*Ashift_rb - θ*Mshift_rb
  rb_rhs = F_rb+H_rb-sum(lifts)
  rb_lhs,rb_rhs
end

function stokes_rb_system(
  lhs::NTuple{2,Matrix{Float}},
  rhs::NTuple{N,Matrix{Float}}) where N

  A_rb,B_rb = lhs
  F_rb,H_rb,lifts... = rhs

  np = size(B_rb)[1]
  rb_lhs = vcat(hcat(A_rb,-B_rb'),hcat(B_rb,zeros(np,np)))
  rb_rhs = vcat(F_rb+H_rb-sum(lifts[1:end-1]),-lifts[end])
  rb_lhs,rb_rhs
end

function stokes_rb_system(
  lhs::NTuple{8,Matrix{Float}},
  rhs::NTuple{N,Matrix{Float}},
  θ::Float) where N

  A_rb,Ashift_rb,BT_rb,BTshift_rb,B_rb,Bshift_rb,M_rb,Mshift_rb = lhs
  F_rb,H_rb,lifts... = rhs

  rb_lhs_11 = θ*(A_rb+M_rb) + (1-θ)*Ashift_rb - θ*Mshift_rb
  rb_lhs_12 = - θ*BT_rb - (1-θ)*BTshift_rb
  rb_lhs_21 = θ*B_rb + (1-θ)*Bshift_rb
  rb_lhs = vcat(hcat(rb_lhs_11,rb_lhs_12),hcat(rb_lhs_21,zeros(np,np)))

  rb_rhs = vcat(F_rb+H_rb-sum(lifts[1:end-1]),-lifts[end])
  rb_lhs,rb_rhs
end

function navier_stokes_rb_system(lhs::Tuple,rhs::Tuple)
  A_rb,B_rb,C_rb,D_rb = lhs
  F_rb,H_rb,lifts = rhs
  liftA,liftB,liftC,_ = lifts

  lin_rb_lhs,lin_rb_rhs = stokes_rb_system([A_rb,B_rb],[F_rb,H_rb,liftA,liftB])

  nu,np = size(A_rb)[1],size(B_rb)[1]
  block12 = zeros(nu,np)
  block21 = zeros(np,nu)
  block22 = zeros(np,np)
  nonlin_rb_lhs1(u) = vcat(hcat(Cₙu(u),block12),hcat(block21,block22))
  nonlin_rb_lhs2(u) = vcat(hcat(C_rb(u)+D_rb(u),block12),hcat(block21,block22))
  nonlin_rb_rhs(u) = vcat(liftC(u),zeros(np,1))

  jac_rb(u) = lin_rb_lhs + nonlin_rb_lhs1(u)
  res_rb(u) = lin_rb_rhs + nonlin_rb_rhs(u)
  dx_rb(u) = jac_rb(u) \ res_rb(u)
  dx_rb
end

function solve_rb_system(rb_lhs::Matrix{Float},rb_rhs::Matrix{Float})
  rb_lhs \ rb_rhs
end

function solve_rb_system(
  V::GridapType,rbspace::Vector{<:RBSpace},dx_rb::Function,args...)
  newton(V,rbspace,dx_rb,args...)
end

function newton(
  rbspace::Vector{<:RBSpace},
  dx_rb::Function,
  ϵ=1e-9,max_iter=10)

  basis_space_u,basis_space_p = get_basis_space.(rbspace)
  Ns_u,ns_u = size(basis_space_u)
  _,ns_p = size(basis_space_p)

  u = FEFunction(V,zeros(Ns_u))
  x_rb = zeros(ns_u+ns_p,1)
  δx̂ = 1. .+ x_rb
  u = FEFunction(V,zeros(Ns_u))
  iter = 0
  err = norm(δx̂)

  while iter < max_iter && err ≥ ϵ
    δx̂ = dx_rb(u,x_rb)
    x_rb -= δx̂
    u = FEFunction(V,basis_space_u*x_rb[1:ns_u])
    iter += 1
    err = norm(δx̂)
    println("Iter: $iter; ||δx̂||₂: $(norm(δx̂))")
  end

  println("Newton-Raphson ended with iter: $iter; ||δx̂||₂: $(norm(δx̂))")
  x_rb
end

function reconstruct_fe_sol(rbspace::RBSpaceSteady,rb_sol::Matrix{Float})
  bs = get_basis_space(rbspace)
  bs*rb_sol
end

function reconstruct_fe_sol(rbspace::RBSpaceUnsteady,rb_sol::Matrix{Float})
  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  ns = get_ns(rbspace)
  nt = get_nt(rbspace)

  rb_sol_resh = reshape(rb_sol,nt,ns)
  bs*(bt*rb_sol_resh)'
end

function reconstruct_fe_sol(rbspace::NTuple{2,RBSpaceSteady},rb_sol::Matrix{Float})
  bs_u,bs_p = get_basis_space.(rbspace)

  ns = get_ns.(rbspace)
  rb_sol_u,rb_sol_p = rb_sol[1:ns[1],:],rb_sol[1+ns[1]:end,:]

  bs_u*rb_sol_u,bs_p*rb_sol_p
end

function reconstruct_fe_sol(rbspace::NTuple{2,RBSpaceUnsteady},rb_sol::Matrix{Float})
  bs_u,bs_p = get_basis_space.(rbspace)
  bt_u,bt_p = get_basis_time.(rbspace)

  ns = get_ns.(rbspace)
  nt = get_nt.(rbspace)
  n = ns.*nt
  rb_sol_u,rb_sol_p = rb_sol[1:n[1],:],rb_sol[1+n[1]:end,:]
  u_rb_resh = reshape(rb_sol_u,nt[1],ns[1])
  p_rb_resh = reshape(rb_sol_p,nt[2],ns[2])

  bs_u*(bt_u*u_rb_resh)',bs_p*(bt_p*p_rb_resh)'
end
