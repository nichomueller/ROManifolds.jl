# INDEX OF TEST
K = 2
μ = realization(feop,K)
times = get_times(fesolver)
Nt = length(times)
N = K*Nt

sols, = collect_solutions(fesolver,feop,get_trial(feop),μ)
nblocks = get_nblocks(sols)

snapsθ = recenter(fesolver,sols,μ)

for nb in 1:nblocks
  [test_ptarray(snapsθ[nb].snaps[i],sols[nb].snaps[i]) for i = eachindex(snapsθ[nb].snaps)]
end

_snaps = vcat(sols[1:K]...)
_snapsθ = vcat(snapsθ[1:K]...)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
Xh, = ode_cache
dxh = ()
_xh = copy(_snaps),_snaps-_snaps
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
b = allocate_residual(ode_op,μ,times,_snapsθ,ode_cache)
vecdata = collect_cell_vector(test,integrate(feop.res(μ,times,xh,dv)))
assemble_vector_add!(b,feop.assem,vecdata)
A = allocate_jacobian(ode_op,μ,times,_snapsθ,ode_cache)
matdata = collect_cell_matrix(trial(μ,times),test,integrate(feop.jacs[1](μ,times,xh,du,dv)))
assemble_matrix_add!(A,feop.assem,matdata)

function gridap_solutions_for_int(n::Int)
  μn = μ[n]

  g_ok(x,t) = g(x,μn,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
  b_ok(t,(v,q)) = ∫(v⋅VectorValue(0.,0.))dΩ
  m_ok(t,(ut,pt),(v,q)) = ∫(ut⋅v)dΩ

  trial_u_ok = TransientTrialFESpace(test_u,g_ok)
  trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  w0 = get_free_dof_values(xh0μ(μn))
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w0,t0,tf)

  sols_ok = []
  for (uh,t) in sol_gridap
    push!(sols_ok,copy(uh))
  end

  sols_ok
end

function nl_gridap_solutions_for_int(n::Int)
  μn = μ[n]

  g_ok(x,t) = g(x,μn,t)
  g_ok(t) = x->g_ok(x,t)
  m_ok(t,u,v) = ∫(v⋅u)dΩ
  a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
  c_ok(t,u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc_ok(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
  jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
  res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,v)

  trial_u_ok = TransientTrialFESpace(test_u,g_ok)
  trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
  feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  w0 = get_free_dof_values(xh0μ(μn))
  nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
  ode_solver = ThetaMethod(nls,dt,θ)
  sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w0,t0,tf)

  sols_ok = []
  for (uh,t) in sol_gridap
    push!(sols_ok,copy(uh))
  end

  sols_ok
end

function gridap_res_jac_for_int(_sols,n::Int)
  μn = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  sols = _sols[fast_idx(n,Nt)]

  g_ok(x,t) = g(x,μn,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
  b_ok(t,(v,q)) = ∫(v⋅VectorValue(0.,0.))dΩ
  m_ok(t,(ut,pt),(v,q)) = ∫(ut⋅v)dΩ

  trial_u_ok = TransientTrialFESpace(test_u,g_ok)
  trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  ode_cache_ok = allocate_cache(ode_op_ok)

  xhF_ok = copy(sols),copy(0. * sols)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = ()
  for i in 2:2
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],xhF_ok[i]))
  end
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],xhF_ok[1]),dxh_ok)
  vecdata_ok = collect_cell_vector(test,feop_ok.res(t,xh_ok,dv))
  res_ok = assemble_vector(feop_ok.assem_t,vecdata_ok)
  matdata_ok = collect_cell_matrix(trial_ok(t),test,feop_ok.jacs[1](t,xh_ok,du,dv))
  jac_ok = assemble_matrix(feop_ok.assem_t,matdata_ok)

  xh_ok,res_ok,jac_ok
end

function nl_gridap_res_jac_for_int(_sols,n::Int)
  μn = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  sols = _sols[fast_idx(n,Nt)]
  sols_prev = fast_idx(n,Nt) == 1 ? zeros(size(sols)) : _sols[fast_idx(n,Nt)-1]
  solsθ = θ*sols + (1-θ)*sols_prev

  g_ok(x,t) = g(x,μn,t)
  g_ok(t) = x->g_ok(x,t)
  m_ok(t,u,v) = ∫(v⋅u)dΩ
  a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
  c_ok(t,u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc_ok(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
  jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
  res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,v)

  trial_u_ok = TransientTrialFESpace(test_u,g_ok)
  trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
  feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  ode_cache_ok = allocate_cache(ode_op_ok)

  # xhF_ok = copy(solsθ),copy(solsθ-sols_prev)/(θ*dt)
  xhF_ok = copy(sols),copy(0. * sols)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = ()
  for i in 2:2
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],xhF_ok[i]))
  end
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],xhF_ok[1]),dxh_ok)
  vecdata_ok = collect_cell_vector(test,feop_ok.res(t,xh_ok,dv))
  res_ok = assemble_vector(feop_ok.assem_t,vecdata_ok)
  matdata_ok = collect_cell_matrix(trial_ok(t),test,feop_ok.jacs[1](t,xh_ok,du,dv))
  jac_ok = assemble_matrix(feop_ok.assem_t,matdata_ok)

  xh_ok,res_ok,jac_ok
end

for np = 1:K
  sols_ok = gridap_solutions_for_int(np)
  for nt in eachindex(sols_ok)
    n = (np-1)*Nt+nt
    @assert isapprox(_snaps[(np-1)*Nt+nt],sols_ok[nt]) "Failed with np = $n, nt = $nt"
    xh_ok,res_ok,jac_ok = gridap_res_jac_for_int(sols_ok,n)
    test_ptarray(b,res_ok;n)
    test_ptarray(A,jac_ok;n)
  end
end

for np = 1:K
  sols_ok = nl_gridap_solutions_for_int(np)
  sols_ok_θ = θ*hcat(sols_ok...) + (1-θ)*hcat(zeros(size(sols_ok[1])),sols_ok[2:end]...)
  for nt in eachindex(sols_ok)
    n = (np-1)*Nt+nt
    @assert isapprox(_snaps[(np-1)*Nt+nt],sols_ok[nt]) "Failed with np = $n, nt = $nt"
    @assert isapprox(_snapsθ[(np-1)*Nt+nt],sols_ok_θ[:,nt]) "Failed with np = $n, nt = $nt"
    xh_ok,res_ok,jac_ok = nl_gridap_res_jac_for_int(sols_ok,n)
    test_ptarray(b,res_ok;n)
    test_ptarray(A,jac_ok;n)
  end
end

# LINEAR
snaps_test,params_test = load_test(info,feop,fesolver)
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
n = 1
un = PTArray(vcat(snaps_test...)[1:Nt])
μn = params_test[n]
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
b_ok(t,(v,q)) = ∫(v⋅VectorValue(0.,0.))dΩ
m_ok(t,(ut,pt),(v,q)) = ∫(ut⋅v)dΩ

trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,Table([μn]),times)
ode_cache = update_cache!(ode_cache,ode_op,Table([μn]),times)
ptb = allocate_residual(ode_op,Table([μn]),times,un,ode_cache)
ptA = allocate_jacobian(ode_op,Table([μn]),times,un,ode_cache)
vθ = copy(un) .* 0.
nlop = get_nonlinear_operator(ode_op,Table([μn]),times,dt*θ,un,ode_cache,vθ)
residual!(ptb,nlop,copy(un))
jacobian!(ptA,nlop,copy(un))
ptb1 = ptb[1:10]
ptA1 = ptA[1:10]

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)/(dt*θ)
vθ = zeros(Nu+Np)
ode_cache = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,vθ,ode_cache,vθ)
bok = allocate_residual(nlop0,vθ)
Aok = allocate_jacobian(nlop0,vθ)

dir(t) = zero(trial_u(μn,t))
ddir(t) = zero(∂ₚt(trial_u)(μn,t))
Lu(t) = assemble_vector(dv->∫(a(μn,t)*∇(dv)⊙∇(dir(t)))dΩ,test_u) + assemble_vector(dv->∫(dv⋅ddir(t))dΩ,test_u)
Lp(t) = -assemble_vector(dq->∫(dq*(∇⋅(dir(t))))dΩ,test_p)

for (kt,t) in enumerate(get_times(fesolver))
  uk = un[kt]
  ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dt*θ,ukprev,ode_cache_ok,vθ)
  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op_ok,t,(vθ,vθ),ode_cache_ok)
  jacobians!(Aok,ode_op_ok,t,(vθ,vθ),(1.0,1/(dt*θ)),ode_cache_ok)
  @assert bok ≈ vcat(Lu(t),Lp(t)) "Failed when n = $kt"
  @assert bok ≈ ptb1[kt] "Failed when n = $kt"
  @assert Aok ≈ ptA1[kt] "Failed when n = $kt"
  bprev = vcat(M*ukprev[1:Nu],zeros(Np))
  @assert Aok \ (bprev - bok) ≈ θ*uk + (1-θ)*ukprev "Failed when n = $kt"
end


#
nblocks = 2
times = get_times(fesolver)
Nt = length(times)
Nu = length(get_free_dof_ids(test_u))
Np = length(get_free_dof_ids(test_p))
# OK
# μ = realization(feop)
# u = [PTArray([zeros(Nu) for _ = 1:Nt]),PTArray([zeros(Np) for _ = 1:Nt])]
snaps_test,params_test = load_test(info,feop,fesolver)
μ = params_test[1]
u = [PTArray(snaps_test[1][1:10]),PTArray(snaps_test[1][1:10])]
vu = vcat(u...)

# RESIDUAL

dir(t) = zero(trial_u(μ,t))
ddir(t) = zero(∂ₚt(trial_u)(μ,t))
Luμ(t) = assemble_vector(dv->∫(a(μ,t)*∇(dv)⊙∇(dir(t)))dΩ,test_u) + assemble_vector(dv->∫(dv⋅ddir(t))dΩ,test_u)
Lpμ(t) = -assemble_vector(dq->∫(dq*(∇⋅(dir(t))))dΩ,test_p)

Rblock = Vector{Matrix{Float}}(undef,nblocks)
Ridx1,Rcomp1 = compress_array(hcat([Luμ(t) for t = times]...))
Ridx2,Rcomp2 = compress_array(hcat([Lpμ(t) for t = times]...))
Rblock[1] = Rcomp1
Rblock[2] = Rcomp2

b1 = PTArray([zeros(Nu) for _ = 1:Nt])
b2 = PTArray([zeros(Np) for _ = 1:Nt])
bblock = Vector{Any}(undef,nblocks)
bblock[1] = b1
bblock[2] = b2
bidx = [Ridx1,Ridx2]

touched = check_touched_residuals(feop,u,Table([μ]),times)
for row = 1:nblocks
  touched_row = touched[row]
  if touched_row
    feop_row_col = feop[row,:]
    b = bblock[row]
    res1, = collect_residuals_for_trian(fesolver,feop_row_col,vu,Table([μ]),times)
    res2 = collect_residuals_for_idx!(b,fesolver,feop_row_col,vu,Table([μ]),times,bidx[row])
    @assert isapprox(res1[1].nonzero_val,Rblock[row])
    @assert isapprox(res2,Rblock[row])
  end
end

# JACOBIAN

Aμ(t) = assemble_matrix((du,dv)->∫(a(μ,t)*∇(dv)⊙∇(du))dΩ,trial_u(μ,t),test_u)
Mμ(t) = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μ,t),test_u)/(dt*θ)
Bμ(t) = -assemble_matrix((du,dq)->∫(dq*(∇⋅(du)))dΩ,trial_u(μ,t),test_p)
Jblock = Matrix{Matrix{Float}}(undef,nblocks,nblocks)
Jidx11 = findnz(Aμ(dt)[:])[1]
Jidx12 = findnz(Bμ(dt)'[:])[1]
Jidx21 = findnz(Bμ(dt)[:])[1]
Aidx = Matrix{Any}(undef,nblocks,nblocks)
Aidx[1,1] = Jidx11
Aidx[1,2] = Jidx12
Aidx[2,1] = Jidx21
Jblock[1,1] = hcat([findnz(Aμ(t)[:])[2] for t = times]...)
Jblock[1,2] = hcat([findnz(Bμ(t)'[:])[2] for t = times]...)
Jblock[2,1] = hcat([findnz(Bμ(t)[:])[2] for t = times]...)

A11 = PTArray([Aμ(dt) for _ = 1:Nt])
fillstored!(A11,0.)
A12 = PTArray([sparse(Bμ(dt)') for _ = 1:Nt])
fillstored!(A12,0.)
A21 = PTArray([Bμ(dt) for _ = 1:Nt])
fillstored!(A21,0.)
Ablock = Matrix{Any}(undef,nblocks,nblocks)
Ablock[1,1] = A11
Ablock[1,2] = A12
Ablock[2,1] = A21

i = 1
touched = check_touched_jacobians(feop,u,Table([μ]),times;i)
for (row,col) = index_pairs(nblocks,nblocks)
  touched_row_col = touched[row,col]
  if touched_row_col
    feop_row_col = feop[row,col]
    snaps_col = u[col]
    A = Ablock[row,col]
    jac1, = collect_jacobians_for_trian(fesolver,feop_row_col,snaps_col,Table([μ]),times;i)
    jac2 = collect_jacobians_for_idx!(A,fesolver,feop_row_col,snaps_col,Table([μ]),times,Aidx[row,col])
    @assert isapprox(jac1[1].nonzero_val,Jblock[row,col])
    @assert isapprox(jac2,Jblock[row,col])
  end
end

# NONLINEAR
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
n = 1
un = PTArray(vcat(snaps_test...)[1:Nt])
μn = params_test[n]
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

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,Table([μn]),times)
ode_cache = update_cache!(ode_cache,ode_op,Table([μn]),times)
ptb = allocate_residual(ode_op,Table([μn]),times,un,ode_cache)
ptA = allocate_jacobian(ode_op,Table([μn]),times,un,ode_cache)
vθ = copy(un) .* 0.
nlop = get_nonlinear_operator(ode_op,Table([μn]),times,dt*θ,un,ode_cache,vθ)
residual!(ptb,nlop,copy(un))
jacobian!(ptA,nlop,copy(un))
ptb1 = ptb[1:Nt]
ptA1 = ptA[1:Nt]

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)/(dt*θ)
vθ = zeros(Nu+Np)
ode_cache = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,vθ,ode_cache,vθ)
bok = allocate_residual(nlop0,vθ)
Aok = allocate_jacobian(nlop0,vθ)

v0 = zeros(size(vθ))

for (kt,t) in enumerate(get_times(fesolver))
  uk = un[kt]
  ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dt*θ,ukprev,ode_cache_ok,vθ)

  x = copy(ukprev)

  for niter in 1:20
    @. vθ = (x-nlop.u0) / nlop.dtθ
    z = zero(eltype(Aok))
    fillstored!(Aok,z)
    fill!(bok,z)
    residual!(bok,ode_op_ok,t,(x,vθ),ode_cache_ok)
    jacobians!(Aok,ode_op_ok,t,(x,vθ),(1.0,1/(dt*θ)),ode_cache_ok) # or v0
    dx = - Aok \ bok
    x .+= dx
    ndx = norm(dx)
    println("Iter $niter, norm error $ndx")
    if ndx ≤ eps()*100
      break
    end
  end

  @assert x ≈ θ*uk + (1-θ)*ukprev "Failed when n = $kt"
end

dir(t) = zero(trial_u(μn,t))
ddir(t) = zero(∂ₚt(trial_u)(μn,t))
Lu_l(u,t) = (assemble_vector(dv->∫(a(μn,t)*∇(dv)⊙∇(u))dΩ,test_u)
  + assemble_vector(dv->∫(dv⋅∂ₚt(u))dΩ,test_u))
Lu_nl(u,t) = assemble_vector(dv->c_ok(t,u,dv),test_u)
Lp(u,t) = -assemble_vector(dq->∫(dq*(∇⋅(u)))dΩ,test_p)
RHS(u,t) = vcat(Lu_nl(u,t) + Lu_l(u,t), Lp(u,t))
RHS_add(u) = vcat(M*u, zeros(Np))
LHS_l_11(t) = M + assemble_matrix((du,dv)->∫(a(μn,t)*∇(dv)⊙∇(du))dΩ,trial_u(μn,t),test_u)
LHS_l_12 = -assemble_matrix((dp,dv)->∫(dp*(∇⋅(dv)))dΩ,trial_p,test_u)
LHS_nl(u,t) = assemble_matrix((du,dv)->dc_ok(t,u,du,dv),trial_u(μn,t),test_u)
LHS(u,t) = vcat(hcat(LHS_l_11(t)+LHS_nl(u,t),LHS_l_12),hcat(LHS_l_12',sparse(zeros(Np,Np))))

for (kt,t) in enumerate(get_times(fesolver))
  uk = un[kt]
  ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dt*θ,ukprev,ode_cache_ok,vθ)

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  jacobians!(Aok,ode_op_ok,t,(ukprev,v0),(1.0,1/(dt*θ)),ode_cache_ok)

  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)

  @assert LHS(xh_ok[1],t) ≈ Aok "Failed when n = $kt"
end

# for (kt,t) in enumerate(get_times(fesolver))
#   uk = un[kt]
#   ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
#   ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
#   nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dt*θ,ukprev,ode_cache_ok,vθ)

#   z = zero(eltype(Aok))
#   fill!(bok,z)
#   residual!(bok,ode_op_ok,t,(ukprev,v0),ode_cache_ok)

#   Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
#   Xh_ok,_,_ = ode_cache_ok
#   dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
#   xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)

#   @assert RHS(xh_ok[1],t) ≈ bok "Failed when n = $kt"
# end

# kt,t = 2,times[2]
# uk = un[kt]
# ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
# ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
# fillstored!(Aok,z)
# fill!(bok,z)
# residual!(bok,ode_op_ok,t,(ukprev,v0),ode_cache_ok)
# Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
# Xh_ok,_,_ = ode_cache_ok
# dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
# xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)

# # bprev = vcat(M*ukprev[1:Nu],zeros(Np))

# RHS(xh_ok[1],t) - bok
for (kt,t) in enumerate(get_times(fesolver))
  uk = un[kt]
  ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dt*θ,ukprev,ode_cache_ok,vθ)

  x = copy(ukprev)

  for niter in 1:20
    @. vθ = (x-nlop.u0) / nlop.dtθ
    z = zero(eltype(Aok))
    fillstored!(Aok,z)
    fill!(bok,z)
    residual!(bok,ode_op_ok,t,(x,vθ),ode_cache_ok)
    jacobians!(Aok,ode_op_ok,t,(x,vθ),(1.0,1/(dt*θ)),ode_cache_ok) # or v0
    dx = - Aok \ bok
    x .+= dx
    ndx = norm(dx)
    println("Iter $niter, norm error $ndx")
    if ndx ≤ eps()*100
      break
    end

    Xh_ok,_,_ = ode_cache_ok
    dxh_ok = (EvaluationFunction(Xh_ok[2],vθ),)
    xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)
    bprev = vcat(M*nlop.u0[1:Nu],zeros(Np))
    @assert RHS(xh_ok[1],t) ≈ bok - bprev
  end

  @assert x ≈ θ*uk + (1-θ)*ukprev "Failed when n = $kt"
end

kt,t = 2,times[2]
uk = un[kt]
ukprev = kt > 1 ? un[kt-1] : get_free_dof_values(xh0μ(μn))
ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dt*θ,ukprev,ode_cache_ok,vθ)

x = copy(ukprev)

niter = 1
@. vθ = (x-nlop.u0) / nlop.dtθ
z = zero(eltype(Aok))
fillstored!(Aok,z)
fill!(bok,z)
residual!(bok,ode_op_ok,t,(x,vθ),ode_cache_ok)
jacobians!(Aok,ode_op_ok,t,(x,vθ),(1.0,1/(dt*θ)),ode_cache_ok) # or v0
dx = - Aok \ bok
x .+= dx
ndx = norm(dx)
println("Iter $niter, norm error $ndx")

Xh_ok,_,_ = ode_cache_ok
dxh_ok = (EvaluationFunction(Xh_ok[2],vθ),)
xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],ukprev),dxh_ok)
bprev = vcat(M*nlop.u0[1:Nu],zeros(Np))
@assert RHS(xh_ok[1],t) + RHS_add(nlop.u0) - LHS(xh_ok[1],t) ≈ bok



ginterp(t) = interpolate_dirichlet(g(μn,t),trial_u(μn,t))
dg_interp(t) = interpolate_dirichlet(∂t(g)(μn,t),trial_u(μn,t))
la(t,v) = ∫(a(μn,t)*∇(v)⊙∇(ginterp(t)))dΩ
lb(t,q) = ∫(q*(∇⋅(ginterp(t))))dΩ
lc(t,u,v) = ∫(v⊙(∇(ginterp(t))'⋅u))dΩ
lm(t,v) = ∫(v⋅dg_interp(t))dΩ
LA(t) = assemble_vector(v->la(t,v),test_u)
LB(t) = assemble_vector(q->lb(t,q),test_p)
LC(t,u) = assemble_vector(v->lc(t,u,v),test_u)
LM(t) = assemble_vector(v->lm(t,v),test_u)

RHS(t,u) = -vcat(LA(t)+LC(t,u)+LM(t),LB(t))
RHSadd(t,uprev) = vcat(M*uprev/(dt*θ),zeros(Np))

Res(t,x,xh,xprev) = LHS(t,x[1])*xh - (RHS(t,x[1])+RHSadd(t,xprev[1:Nu]))
