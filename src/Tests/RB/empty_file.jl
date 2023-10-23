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
z = zero(eltype(Aok))
fillstored!(Aok,z)
fill!(bcopy,z)
residual!(bcopy,ode_op_ok,t,(x,v0),ode_cache_ok)
bprev = vcat(M*vθ[1:Nu],zeros(Np))
jacobians!(Aok,ode_op_ok,t,(x,v0),(1.0,1/(dt*θ)),ode_cache_ok)

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u) / (θ*dt)
AA(t) = assemble_matrix((du,dv)->∫(a(μn,t)*∇(dv)⊙∇(du))dΩ,trial_u(μn,t),test_u)
dC(u,t) = assemble_matrix((du,dv)->dc_ok(t,u,du,dv),trial_u(μn,t),test_u)
B = -assemble_matrix((du,dq)->∫(dq*(∇⋅(du)))dΩ,trial_u(μn,dt),test_p)
R1((u,p),t) = (assemble_vector(dv -> ∫(dv⋅∂ₚt(u))dΩ,test_u)
  + assemble_vector(dv -> ∫(a(μn,t)*∇(dv)⊙∇(u))dΩ,test_u)
  + assemble_vector(dv -> c_ok(t,u,dv),test_u)
  - assemble_vector(dv -> ∫(p*(∇⋅(dv)))dΩ,test_u))
R2((u,p),t) = - assemble_vector(dq -> ∫(dq*(∇⋅(u)))dΩ,test_p)

function my_get_u(u,t)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  TransientCellField(EvaluationFunction(Xh_ok[1],u),dxh_ok)
end

xn0 = map(zero,xn)

for iter in 1:fesolver.nls.max_nliters
  xtest = vcat(xn0...)

  LHS111 = NnzMatrix([NnzVector(AA(t) + dC(my_get_u(u,t)[1],t)) for (u,t) in zip(xtest.array,times)]...)
  LHS112 = NnzMatrix([NnzVector(M) for _ = 1:ntimes]...)
  LHS21 = NnzMatrix([NnzVector(B) for _ = 1:ntimes]...)
  LHS12 = NnzMatrix([NnzVector(sparse(B')) for _ = 1:ntimes]...)

  LHS111_rb = space_time_projection(LHS111,rbspace[1],rbspace[1])
  LHS112_rb = space_time_projection(LHS112,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS111_rb + LHS112_rb
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

  np = get_rb_ndofs(rbspace[2])
  LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

  R11 = NnzMatrix([R1(my_get_u(u,t),t) for (u,t) in zip(xtest.array,times)]...)
  R21 = NnzMatrix([R2(my_get_u(u,t),t) for (u,t) in zip(xtest.array,times)]...)
  RHS1_rb = space_time_projection(R11,rbspace[1])
  RHS2_rb = space_time_projection(R21,rbspace[2])
  RHS_rb = vcat(RHS1_rb,RHS2_rb)

  # println(norm(LHS_rb))
  # println(norm(RHS_rb))
  dxrb = NonaffinePTArray([LHS_rb \ RHS_rb])
  xn0 -= recast(dxrb,rbspace)

  println("norm dx = $(norm(map(norm,dxrb.array)))")
end

xappcat = hcat(vcat(xn0...).array...)
xcat = hcat(vcat(xn...).array...)
norm(xcat - xappcat)

nu = get_rb_ndofs(rbspace[1])
xn0 = map(zero,xn)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,Table([μn]))
for iter in 1:fesolver.nls.max_nliters
  xtest = vcat(xn0...)
  LHS111 = NnzMatrix([NnzVector(AA(t) + dC(my_get_u(u,t)[1],t)) for (u,t) in zip(xtest.array,times)]...)
  LHS112 = NnzMatrix([NnzVector(M) for _ = 1:ntimes]...)
  LHS21 = NnzMatrix([NnzVector(B) for _ = 1:ntimes]...)
  LHS12 = NnzMatrix([NnzVector(sparse(B')) for _ = 1:ntimes]...)
  LHS111_rb = space_time_projection(LHS111,rbspace[1],rbspace[1])
  LHS112_rb = space_time_projection(LHS112,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS111_rb + LHS112_rb
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])
  np = get_rb_ndofs(rbspace[2])
  LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))
  R11 = NnzMatrix([R1(my_get_u(u,t),t) for (u,t) in zip(xtest.array,times)]...)
  R21 = NnzMatrix([R2(my_get_u(u,t),t) for (u,t) in zip(xtest.array,times)]...)
  RHS1_rb = space_time_projection(R11,rbspace[1])
  RHS2_rb = space_time_projection(R21,rbspace[2])
  RHS_rb = vcat(RHS1_rb,RHS2_rb)
  # dxrb = NonaffinePTArray([LHS_rb \ RHS_rb])
  # xn0 -= recast(dxrb,rbspace)

  rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,xn0,Table([μn]))
  lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,xn0,Table([μn]))

  dxrb = NonaffinePTArray([lhs[1] \ RHS_rb]) # vcat(RHS1_rb,rhs[1][1+nu:end])])#
  xn0 -= recast(dxrb,rbspace)
  # println(norm(LHS_rb - lhs[1]))
  # println(norm(RHS_rb - rhs[1]))

  println("norm dx = $(norm(map(norm,dxrb.array)))")
end

# CONCLUSIONS: THE RESIDUAL BASIS DOESN'T DESCRIBE THE MANIFOLD

# case 1 works
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]

# case 2 doesn't
xn,μn = zero.([PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])]),params_test[1:1]

# case 3
xn,μn = [PTArray(snaps_test[1][1+ntimes:2*ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]

_xn = vcat(xn...)
rbrest = rbrhs[1][Ω]
_feop = feop[1,:]
offsets = field_offsets(feop.test)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,μn)
rhs_cache_row = cache_at_index(rhs_cache,offsets[1]+1:offsets[2])
rcache,scache... = rhs_cache[1]

times = get_times(fesolver)
red_idx = rbrest.integration_domain.idx
red_times = rbrest.integration_domain.times
red_meas = rbrest.integration_domain.meas
full_idx = collect(get_free_dof_ids(_feop.test))

b = PTArray(rcache[1:length(red_times)*length(μn)])
_xn = get_solutions_at_times(_xn,fesolver,red_times)
bfull = copy(b)
Res_full = collect_residuals_for_idx!(bfull,fesolver,_feop,_xn,μn,red_times,full_idx,dΩ)
Res_offline,trian = collect_residuals_for_trian(fesolver,_feop,vcat(sols[1:20]...),params[1:20],times)

_basis_space = tpod(Res_offline[1])
err_proj = Res_full - _basis_space*_basis_space'*Res_full

#
row,col = 1,1
rbjact = rblhs[1][row,col][Ω]
_feop = feop[row,col]
_xn = xn[1]

lhs_cache_row_col = cache_at_index(lhs_cache,offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1])
jcache,scache... = lhs_cache_row_col[1]

times = get_times(fesolver)
red_idx = rbjact.integration_domain.idx
red_times = rbjact.integration_domain.times
red_meas = rbjact.integration_domain.meas


A = PTArray(jcache[1:length(red_times)*length(μn)])
_xn = get_solutions_at_times(_xn,fesolver,red_times)
Afull = copy(A)
full_idx = Afull[1][:].nzind
Jac_full = collect_jacobians_for_idx!(Afull,fesolver,_feop,_xn,μn,red_times,full_idx,dΩ)
Jac_offline,trian = collect_jacobians_for_trian(fesolver,_feop,vcat(sols[1:20]...),params[1:20],times)

_basis_space = tpod(Jac_offline[1])
err_proj = Jac_full - _basis_space*_basis_space'*Jac_full


#
# alternative
_jac(μ,t,(u,p),(du,dp),(v,q)) = a(μ,t,(du,dp),(v,q)) + ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
_feop = PTFEOperator(res,_jac,jac_t,pspace,trial,test)

# OFFLINE
row,col = 1,1
nsnaps = info.nsnaps_system
snapsθ = recenter(fesolver,sols,params)
_snapsθ,_μ = snapsθ[1:nsnaps],params[1:nsnaps]

# block (1,1) _jac
ad_lhs11 = collect_compress_lhs(info,_feop[1,1],fesolver,rbspace[1],_snapsθ[1],_μ)
# _res
ad_rhs = collect_compress_rhs(info,_feop,fesolver,rbspace,zero.(_snapsθ),_μ)

# ONLINE
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,μn)
x = map(zero,xn)
x0 = map(copy,x)
offsets = field_offsets(feop.test)
lhs_cache_row_col = cache_at_index(lhs_cache,offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1])

for iter in 1:fesolver.nls.max_nliters
  j = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,μn)
  lhs11 = collect_lhs_contributions!(lhs_cache_row_col,info,_feop[1,1],fesolver,ad_lhs11,rbspace[1],x[1],μn)
  lhs = vcat(hcat(lhs11[1],j[1][1:nu,1+nu:end]),j[1][1+nu:end,:])
  rhs = collect_rhs_contributions!(rhs_cache,info,_feop,fesolver,ad_rhs,rbspace,x0,μn)
  xrb = space_time_projection(x,rbspace)
  r = lhs*xrb[1] - rhs[1]

  dxrb = NonaffinePTArray([j[1] \ r])
  x -= recast(dxrb,rbspace)
  err = map(norm,dxrb.array)
  err_inf = norm(err)
  println("Iter $iter, error norm: $err_inf")
end

xappcat = hcat(vcat(x...).array...)
xcat = hcat(vcat(xn...).array...)
norm(xcat - xappcat)
