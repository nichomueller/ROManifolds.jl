times = get_times(fesolver)
ntimes = length(times)
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1]
xcat = vcat(xn...)

g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
Jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
Res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,v)
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

function return_quantities(ode_cache_ok,x::PTArray,kt)
  xk = x[kt]
  xk_1 = kt > 1 ? x[kt-1] : get_free_dof_values(xh0μ(μn))
  t = times[kt]
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)

  vθ = (xk-xk_1) / θdt
  bprev = vcat(M*vθ[1:Nu],zeros(Np))

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op_ok,t,(xk,v0),ode_cache_ok)
  jacobians!(Aok,ode_op_ok,t,(xk,v0),(1.0,1/(dt*θ)),ode_cache_ok)
  return (bok,bprev),Aok
end

x = copy(xcat) .* 0.
nu = get_rb_ndofs(rbspace[1])
# WORKS!!!!!!!!!!
for iter in 1:fesolver.nls.max_nliters
  A,b = [],[]
  for kt = eachindex(times)
    (bk,_),Ak = return_quantities(ode_cache_ok,x,kt)
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
for fun in (:(Algebra.residual!),:residual_for_trian!)
  @eval begin
    function $fun(
      b::PTArray,
      op::PThetaMethodNonlinearOperator,
      x::PTArray,
      args...)

      println("DIO CANE")
      uθ = zero(x)
      vθ = zero(x)
      z = zero(eltype(b))
      fill!(b,z)
      $fun(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache,args...)
    end
  end
end

Base.adjoint(::Nothing) = nothing
_c(u,du,dv) = ∫ₚ(dv⊙(∇(du)'⋅u),dΩ)
_jac(μ,t,(u,p),(du,dp),(v,q)) = a(μ,t,(du,dp),(v,q)) + _c(u,du,v)
_feop = PTFEOperator(res,_jac,jac_t,pspace,trial,test)

rbrhs,rbjac = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
_,rblhs = collect_compress_rhs_lhs(info,_feop,fesolver,rbspace,sols,params)

rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,Table([μn]))

y = map(zero,xn)
ycat = vcat(y...)
yrb = space_time_projection(map(x->x[1:Nu],ycat),rbspace[1]),space_time_projection(map(x->x[1+Nu:end],ycat),rbspace[2])
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,y,Table([μn]))
j = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rbjac,rbspace,y,Table([μn]))
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,y,Table([μn]))
r = lhs*vcat(yrb...) + rhs

x = copy(xcat) .* 0.
A,b = [],[]
for kt = eachindex(times)
  (bk,_),Ak = return_quantities(ode_cache_ok,x,kt)
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
println("Error norm (jac,res): ($(norm(lhs[1] - LHS_rb)),$(norm(rhs[1] - RHS_rb)))")

y = map(zero,xn)
for iter in 1:fesolver.nls.max_nliters
  ycat = vcat(y...)
  yrb = space_time_projection(map(x->x[1:Nu],ycat),rbspace[1]),space_time_projection(map(x->x[1+Nu:end],ycat),rbspace[2])
  rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,y,Table([μn]))
  j = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rbjac,rbspace,y,Table([μn]))
  lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,y,Table([μn]))
  r = lhs*vcat(yrb...) + rhs

  dxrb = NonaffinePTArray([j[1] \ r[1]])
  y -= recast(dxrb,rbspace)

  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xcat.array...) - hcat(vcat(y...).array...)) / norm(hcat(xcat.array...))
