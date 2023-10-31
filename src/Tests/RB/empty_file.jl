times = get_times(fesolver)
ntimes = length(times)
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1]
xcat = vcat(xn...)

g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ + ∫(v⊙(∇(u)'⋅du))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
Jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
Res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,u,v)
trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(Res_ok,Jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)

times = get_times(fesolver)
kt = 1
t = times[kt]
v0 = zero(xcat[1])
x = kt > 1 ? xcat[kt-1] : get_free_dof_values(xh0μ(μn))
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
Nt = get_time_ndofs(fesolver)
θdt = θ*dt
vθ = zeros(Nu+Np)
ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,vθ,ode_cache_ok,vθ)
bok = allocate_residual(nlop0,vθ)
Aok = allocate_jacobian(nlop0,vθ)

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)

function return_quantities(ode_op,ode_cache,x::PTArray,kt)
  xk = x[kt]
  xk_1 = kt > 1 ? x[kt-1] : get_free_dof_values(xh0μ(μn))
  t = times[kt]
  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,ode_op,t)

  vθ = (xk-xk_1) / θdt
  bprev = vcat(M*vθ[1:Nu],zeros(Np))

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op,t,(xk,v0),ode_cache)
  jacobians!(Aok,ode_op,t,(xk,v0),(1.0,1/(dt*θ)),ode_cache)
  return (bok,bprev),Aok
end

x = copy(xcat) .* 0.
nu = get_rb_ndofs(rbspace[1])
# WORKS!!!!!!!!!!
for iter in 1:fesolver.nls.max_nliters
  A,b = [],[]
  for kt = eachindex(times)
    (bk,_),Ak = return_quantities(ode_op_ok,ode_cache_ok,x,kt)
    push!(b,copy(bk))
    push!(A,copy(Ak))
  end

  LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/θdt),A)...)
  LHS11_2 = NnzMatrix([NnzVector(M/θdt) for _ = times]...)
  LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
  LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

  LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
  LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS11_rb_1 + LHS11_rb_2
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

  np = get_rb_ndofs(rbspace[2])
  LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

  R1 = NnzMatrix(map(x->x[1:Nu],b)...)
  R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
  RHS1_rb = space_time_projection(R1,rbspace[1])
  RHS2_rb = space_time_projection(R2,rbspace[2])
  RHS_rb = vcat(RHS1_rb,RHS2_rb)

  println("Norm (RHS1,LHS11) = ($(RHS1_rb[1]),$(norm(LHS11_rb[1])))")
  xrb = space_time_projection(map(x->x[1:Nu],x),rbspace[1]),space_time_projection(map(x->x[1+Nu:end],x),rbspace[2])
  dxrb = LHS_rb \ (vcat(LHS11_rb_2*vcat(xrb...)[1][1:nu],zeros(np)) + RHS_rb)
  dxrb_1,dxrb_2 = dxrb[1:nu],dxrb[1+nu:end]
  xiter = vcat(recast(PTArray(dxrb_1),rbspace[1]),recast(PTArray(dxrb_2),rbspace[2]))
  x -= xiter

  nerr = norm(dxrb)
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xcat.array...) - hcat(x.array...)) / norm(hcat(xcat.array...))

# COMPARISON
x = copy(xcat) #.* 0.
A,b = [],[]
for kt = eachindex(times)
  (bk,_),Ak = return_quantities(ode_op_ok,ode_cache_ok,x,kt)
  push!(b,copy(bk))
  push!(A,copy(Ak))
end

LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/θdt),A)...)
LHS11_2 = NnzMatrix([NnzVector(M/θdt) for _ = times]...)
LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
LHS11_rb = LHS11_rb_1 + LHS11_rb_2
LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

np = get_rb_ndofs(rbspace[2])
LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

R1 = NnzMatrix(map(x->x[1:Nu],b)...)
R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
RHS1_rb = space_time_projection(R1,rbspace[1])
RHS2_rb = space_time_projection(R2,rbspace[2])
RHS_rb = vcat(RHS1_rb,RHS2_rb)

op = get_ptoperator(fesolver,feop,x,Table([μn]))
xrb = space_time_projection(x,op,rbspace)
rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
llhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
nllhs = collect_lhs_contributions!(lhs_cache,info,op,nl_rblhs,rbspace)
_LHS_rb = llhs + nllhs
_RHS_rb = rhs - llhs*xrb
norm(LHS_rb - _LHS_rb[1],Inf)
norm(RHS_rb + _RHS_rb[1],Inf)

# verify splitting operator
x = copy(xcat)

A,b = [],[]
for kt = eachindex(times)
  (bk,_),Ak = return_quantities(ode_op_ok,ode_cache_ok,x,kt)
  push!(b,copy(bk))
  push!(A,copy(Ak))
end

Jac_ok_lin(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + c_ok(t,u,du,v)
Res_ok_lin(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,u,v)
Jac_ok_nlin(t,(u,p),(du,dp),(v,q)) = ∫(v⊙(∇(u)'⋅du))dΩ
Res_ok_nlin(t,(u,p),(v,q)) = nothing
jac_t_ok_nlin(t,(u,p),(du,dp),(v,q)) = nothing

feop_ok_lin = TransientFEOperator(Res_ok_lin,Jac_ok_lin,jac_t_ok,trial_ok,test)
ode_op_ok_lin = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok_lin)
ode_cache_ok_lin = allocate_cache(ode_op_ok_lin)

Alin,blin = [],[]
for kt = eachindex(times)
  (bk,_),Ak = return_quantities(ode_op_ok_lin,ode_cache_ok_lin,x,kt)
  push!(blin,copy(bk))
  push!(Alin,copy(Ak))
end

feop_ok_nlin = TransientFEOperator(Res_ok_nlin,Jac_ok_nlin,jac_t_ok_nlin,trial_ok,test)
ode_op_ok_nlin = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok_nlin)
ode_cache_ok_nlin = allocate_cache(ode_op_ok_nlin)


FESpaces.numeric_loop_vector!(b,a::SparseMatrixAssembler,vecdata::Nothing) = nothing
function Gridap.ODEs.TransientFETools.fill_jacobians(
  op::Gridap.ODEs.TransientFETools.TransientFEOperatorsFromWeakForm,
  t::Real,
  xh::T,
  γ::Tuple{Vararg{Real}}) where T
  _matdata = ()
  for i in 1:Gridap.ODEs.TransientFETools.get_order(op)+1
    if (γ[i] > 0.0)
      _data = Gridap.ODEs.TransientFETools._matdata_jacobian(op,t,xh,i,γ[i])
      if !isnothing(_data)
        _matdata = (_matdata...,_data)
      end
    end
  end
  return _matdata
end

Anlin,bnlin = [],[]
for kt = eachindex(times)
  (bk,_),Ak = return_quantities(ode_op_ok_nlin,ode_cache_ok_nlin,x,kt)
  push!(bnlin,copy(bk))
  push!(Anlin,copy(Ak))
end

for kt = eachindex(times)
  @assert A[kt] ≈ Alin[kt] + Anlin[kt] "Jac failed when kt = $kt"
  @assert b[kt] ≈ blin[kt] + bnlin[kt] "Res failed when kt = $kt"
end

# linear part
LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/θdt),Alin)...)
LHS11_2 = NnzMatrix([NnzVector(M/θdt) for _ = times]...)
LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),Alin)...)
LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),Alin)...)

LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
LHS11_rb = LHS11_rb_1 + LHS11_rb_2
LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

np = get_rb_ndofs(rbspace[2])
LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

R1 = NnzMatrix(map(x->x[1:Nu],blin)...)
R2 = NnzMatrix(map(x->x[1+Nu:end],blin)...)
RHS1_rb = space_time_projection(R1,rbspace[1])
RHS2_rb = space_time_projection(R2,rbspace[2])
RHS_rb = vcat(RHS1_rb,RHS2_rb)

op = get_ptoperator(fesolver,feop,x,Table([μn]))
xrb = space_time_projection(x,op,rbspace)
rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
llhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
_LHS_rb = llhs + nllhs
_RHS_rb = rhs - llhs*xrb
norm(LHS_rb - _LHS_rb[1],Inf)
norm(RHS_rb + _RHS_rb[1],Inf)

#nonlinear part
LHS11 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu]),Anlin)...)
LHS11_rb = space_time_projection(LHS11,rbspace[1],rbspace[1])

nu = get_rb_ndofs(rbspace[1])
np = get_rb_ndofs(rbspace[2])
LHS_rb = vcat(hcat(LHS11_rb,zeros(nu,np)),hcat(zeros(np,nu),zeros(np,np)))

op = get_ptoperator(fesolver,feop,x,Table([μn]))
xrb = space_time_projection(x,op,rbspace)
nllhs = collect_lhs_contributions!(lhs_cache,info,op,nl_rblhs,rbspace)
norm(LHS_rb - nllhs[1],Inf)

#
times = get_times(fesolver)
Nt = length(times)
Nu = length(get_free_dof_ids(test_u))
Np = length(get_free_dof_ids(test_p))
array = [rand(Nu+Np) for _ = 1:Nt]
un = PTArray(vcat(snaps_test...)[1:Nt])
μn = params_test[n]

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ
c(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ
dir(t) = zero(trial_u(μn,t))
liftc(t,u,v) = c(t,u,dir(t),v)

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
Jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
Res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,v)

trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(Res_ok,Jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)
times = get_times(fesolver)
dv = get_fe_basis(test)
v0 = zeros(Nu+Np)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,v0,ode_cache_ok,v0)
bok = allocate_residual(nlop0,v0)
b = allocate_residual(nlop0,v0)
A = allocate_jacobian(nlop0,v0)

for (kt,t) in enumerate(times)
  uk = un[kt]
  ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)
  xh0_ok = TransientCellField(EvaluationFunction(Xh_ok[1],v0),dxh_ok)

  # WORKS
  # r1 = assemble_vector(dv->a_ok(t,xh_ok,dv),test)
  # r2 = (assemble_vector(dv->a_ok(t,xh0_ok,dv),test)
  #   + assemble_matrix((du,dv)->a_ok(t,du,dv),trial_ok(t),test)*ukprev)

  # WORKS
  # r1 = assemble_vector(dv->m_ok(t,xh_ok[1],dv),test[1])
  # r2 = (assemble_vector(dv->m_ok(t,xh0_ok[1],dv),test[1])
  #   + assemble_matrix((du,dv)->m_ok(t,du,dv),trial_ok(t)[1],test[1])*ukprev[1:Nu])

  # WORKS
  # r1 = assemble_vector(dv->c_ok(t,xh_ok[1],dv),test[1])
  # r2 = (assemble_vector(dv->liftc(t,xh_ok[1],dv),test[1])
  #   + assemble_matrix((du,dv)->c(t,xh_ok[1],du,dv),trial_ok(t)[1],test[1])*ukprev[1:Nu])
  # @assert isapprox(r1,r2) "Failed when kt = $kt"

  # WORKS
  r1 = assemble_vector(dv->c_ok(t,xh_ok[1],dv),test[1])
  r2 = (assemble_vector(dv->c(t,xh_ok[1],xh0_ok[1],dv),test[1])
    + assemble_matrix((du,dv)->c(t,xh_ok[1],du,dv),trial_ok(t)[1],test[1])*ukprev[1:Nu])
  @assert isapprox(r1,r2) "Failed when kt = $kt"
end

# doesnt work
for (kt,t) in enumerate(times)
  uk = un[kt]
  ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)
  xh0_ok = TransientCellField(EvaluationFunction(Xh_ok[1],v0),dxh_ok)

  z = zero(eltype(bok))
  fill!(bok,z)
  residual!(bok,ode_op_ok,t,(ukprev,v0),ode_cache_ok)
  fill!(b,z)
  residual!(b,ode_op_ok,t,(v0,v0),ode_cache_ok)
  fillstored!(A,z)
  jacobians!(A,ode_op_ok,t,(v0,v0),(1.0,1/(dt*θ)),ode_cache_ok)

  vec_from_mat = A*ukprev
  bnok = b+vec_from_mat

  @assert isapprox(bok,bnok) "Failed when kt = $kt"
end
