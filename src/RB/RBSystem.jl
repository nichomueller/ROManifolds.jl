#= function assemble_rb_system(
  op::RBVarOperator,
  basis::NTuple{2,Matrix{Float}},
  μ::Param,
  args...)

  Broadcasting(b->assemble_rb_system(op,b,μ,args...))(basis)
end =#

function assemble_rb_system(
  op::NTuple{N,<:RBVarOperator},
  rbvar::Tuple,
  args...) where N

  Broadcasting((o,rbv)->assemble_rb_system(o,rbv,args...))(op,rbvar)
end

function assemble_rb_system(
  op::RBVarOperator,
  basis::Matrix{Float},
  μ::Param,
  args...)

  coeff = compute_coefficient(op,μ)
  assemble_rb_system(op,basis,coeff)
end

#= function assemble_rb_system(
  op::RBVarOperator,
  mdeim::Union{MDEIM,NTuple{2,<:MDEIM}},
  μ::Param,
  args...)

  coeff = compute_coefficient(op,mdeim,μ,args...)
  basis = get_basis_space(mdeim)
  assemble_rb_system(op,basis,coeff,args...)
end =#

function assemble_rb_system(
  op::RBVarOperator,
  mdeim::MDEIM,
  μ::Param,
  args...)

  coeff = compute_coefficient(op,mdeim,μ,args...)
  basis = get_basis_space(mdeim)
  assemble_rb_system(op,basis,coeff)
end

function assemble_rb_system(
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  basis::Matrix{Float},
  coeff::Array{Float})

  nr = get_nrows(op)
  basis_by_coeff_mult(basis,coeff,nr)
end

function assemble_rb_system(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  basis::Matrix{Float},
  coeff::Array{Float})

  btbtp = multiply_time_bases(op,coeff)
  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff /= dtθ end
  nr = get_nrows(op)
  basis_by_coeff_mult(basis,btbtp,nr)
end

function assemble_rb_system(
  op::Union{RBSteadyLinOperator{Nonlinear},RBSteadyBilinOperator{Nonlinear,TT},RBSteadyLiftingOperator{Nonlinear,TT}},
  basis::Matrix{Float},
  coeff::Array{Function}) where TT

  nr = get_nrows(op)
  u -> basis_by_coeff_mult(basis,coeff(u),nr)
end

function assemble_rb_system(
  op::Union{RBUnsteadyLinOperator{Nonlinear},RBUnsteadyBilinOperator{Nonlinear,TT},RBUnsteadyLiftingOperator{Nonlinear,TT}},
  basis::Matrix{Float},
  coeff::Array{Function}) where TT

  btbtp = multiply_time_bases(op,coeff)
  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff /= dtθ end
  nr = get_nrows(op)
  u -> basis_by_coeff_mult(basis,btbtp(u),nr)
end

function multiply_time_bases(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyLiftingOperator},
  coeff::AbstractMatrix)

  bt_row = get_basis_time_row(op)
  nt_row = size(bt_row,2)
  Q = size(coeff,1)

  btp_fun(it,q) = sum(bt_row[:,it].*coeff[q,:])
  btp_fun(q) = Broadcasting(it -> btp_fun(it,q))(1:nt_row)
  Matrix(reshape(Matrix(btp_fun.(1:Q)),:,Q))
end

function multiply_time_bases(
  op::RBUnsteadyBilinOperator,
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
    btbtp_fun(q) = Matrix(Broadcasting(jt -> btbtp_fun(jt,q))(1:nt_col))[:]
    btbtp_block = Matrix(btbtp_fun.(1:Q))
    btbtp_block
  end

  M = define_btbtp_fun(idx,idx)
  M_shift = define_btbtp_fun(idx_shift_forward,idx_shift_backward)
  M,M_shift
end

#= function multiply_time_bases(
  op::RBBilinOperator{Top,TT,RBSpaceUnsteady},
  coeff::NTuple{2,AbstractMatrix}) where {Top,TT}

  c,clift = coeff
  M,M_shift = multiply_time_bases(op,c)

  bt_row = get_basis_time_row(op)
  nt_row = size(bt_row)[2]
  btp_fun(it,q) = sum(bt_row[:,it].*clift[q])
  btp_fun(q) = reshape(Broadcasting(it -> btp_fun(it,q))(1:nt_row),:,1)
  M_lift = Matrix(btp_fun.(eachindex(clift)))

  M,M_shift,M_lift
end =#

function multiply_time_bases(
  op::RBUnsteadyBilinOperator,
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
    btbtp_fun(u,q) = Matrix(Broadcasting(jt -> btbtp_fun(u,jt,q))(1:nt_col))[:]
    btbtp_fun(u) = Matrix(Broadcasting(q -> btbtp_fun(u,q))(1:Q))
    btbtp_fun
  end

  M = define_btbtp_fun(idx,idx)
  M_shift = define_btbtp_fun(idx_shift_forward,idx_shift_back)
  M,M_shift
end

#= function multiply_time_bases(
  op::RBBilinOperator{Top,TT,RBSpaceUnsteady},
  coeff::NTuple{2,Function}) where {Top,TT}

  c(u) = first(coeff)(u)
  M,M_shift = multiply_time_bases(op,c)

  clift(u) = last(coeff)(u)
  bt_row = get_basis_time_row(op)
  nt_row = size(bt_row)[2]
  btp_fun(u,it,q) = sum(bt_row[:,it].*clift(u)[q])
  btp_fun(u,q) = reshape(Broadcasting(it -> btp_fun(u,it,q))(1:nt_row),:,1)
  Mlift(u) = Matrix(btp_fun.(eachindex(clift(u))))

  M,M_shift,Mlift
end =#

function poisson_rb_system(lhs::Matrix{Float},rhs::Vector{Matrix{Float}})
  A_rb = lhs
  F_rb,H_rb,lifts = rhs
  A_rb,F_rb+H_rb-sum(lifts,dims=2)
end

function poisson_rb_system(lhs::Vector{Matrix{Float}},rhs::Vector{Matrix{Float}},θ::Float)
  A_rb,Ashift_rb,M_rb,Mshift_rb = lhs
  F_rb,H_rb,lifts = rhs

  rb_lhs = θ*(A_rb+M_rb) + (1-θ)*Ashift_rb - θ*Mshift_rb
  rb_rhs = F_rb+H_rb-sum(lifts,dims=2)
  rb_lhs,rb_rhs
end

function stokes_rb_system(lhs::Vector{Matrix{Float}},rhs::Vector{Matrix{Float}})
  A_rb,B_rb = lhs
  F_rb,H_rb,lifts = rhs

  np = size(B_rb)[1]
  rb_lhs = vcat(hcat(A_rb,-B_rb'),hcat(B_rb,zeros(np,np)))
  rb_rhs = vcat(F_rb+H_rb-sum(lifts[1:end-1],dims=2),-lifts[end])
  rb_lhs,rb_rhs
end

function stokes_rb_system(lhs::Vector{Matrix{Float}},rhs::Vector{Matrix{Float}},θ::Float)
  A_rb,Ashift_rb,BT_rb,BTshift_rb,B_rb,Bshift_rb,M_rb,Mshift_rb = lhs
  F_rb,H_rb,lifts = rhs

  rb_lhs_11 = θ*(A_rb+M_rb) + (1-θ)*Ashift_rb - θ*Mshift_rb
  rb_lhs_12 = - θ*BT_rb - (1-θ)*BTshift_rb
  rb_lhs_21 = θ*B_rb + (1-θ)*Bshift_rb
  rb_lhs = vcat(hcat(rb_lhs_11,rb_lhs_12),hcat(rb_lhs_21,zeros(np,np)))

  rb_rhs = vcat(F_rb+H_rb-sum(lifts[1:end-1],dims=2),-lifts[end])
  rb_lhs,rb_rhs
end

function navier_stokes_rb_system(lhs::Vector,rhs::Vector{Matrix{Float}})
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

function reconstruct_fe_sol(bspace::RBSpaceSteady,rb_sol::Matrix{Float})
  bs = get_basis_space(bspace)
  bs*rb_sol
end

function reconstruct_fe_sol(bspace::RBSpaceUnsteady,rb_sol::Matrix{Float})
  bs = get_basis_space(bspace)
  bt = get_basis_time(bspace)
  ns = get_ns(bspace)
  nt = get_nt(bspace)

  rb_sol_resh = reshape(rb_sol,nt,ns)
  bs*(bt*rb_sol_resh)'
end
