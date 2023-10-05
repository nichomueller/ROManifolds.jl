# INDEX OF TEST
N = 10

function pt_quantities(N,trian)
  # SOL
  μ = realization(feop,N)
  sols = collect_solutions(fesolver,feop,μ)

  # RES/JAC
  times = get_times(fesolver)
  snapsθ = recenter(fesolver,sols,μ)
  [test_ptarray(snapsθ.snaps[i],sols.snaps[i]) for i = eachindex(snapsθ.snaps)]
  _μ,_snapsθ = μ[1:N],snapsθ[1:N]
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,_μ,times)
  ode_cache = update_cache!(ode_cache,ode_op,_μ,times)
  Xh, = ode_cache
  dxh = ()
  _xh = (_snapsθ,_snapsθ-_snapsθ)
  for i in 2:get_order(feop)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
  end
  xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing,nothing))

  b = allocate_residual(ode_op,_μ,times,_snapsθ,ode_cache)
  vecdata = collect_cell_vector(test,feop.res(_μ,times,xh,dv),trian)
  assemble_vector_add!(b,feop.assem,vecdata)
  A = allocate_jacobian(ode_op,_μ,times,_snapsθ,ode_cache)
  matdata = collect_cell_matrix(trial(_μ,times),test,feop.jacs[1](_μ,times,xh,du,dv),trian)
  assemble_matrix_add!(A,feop.assem,matdata)
  sols,A,b
end

function gridap_quantities_for_param(n::Int,trian::Triangulation)
  p = μ[n]
  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
  b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
  m_ok(t,ut,v) = ∫(ut*v)dΩ

  trial_ok = TransientTrialFESpace(test,g_ok)
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  w0 = get_free_dof_values(uh0μ(p))
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w0,t0,tf)

  sols_ok = []
  for (uh,t) in sol_gridap
    push!(sols_ok,copy(uh))
  end

  res_ok,jac_ok = [],[]
  for (n,tn) in enumerate(times)
    xhF_ok = copy(_snapsθ[n]),0. * _snapsθ[n]
    Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,tn)
    Xh_ok,_,_ = ode_cache_ok
    dxh_ok = ()
    for i in 2:2
      dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],xhF_ok[i]))
    end
    xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],xhF_ok[1]),dxh_ok)
    vecdata_ok = collect_cell_vector(test,feop_ok.res(tn,xh_ok,dv),trian)
    push!(res_ok,assemble_vector(feop_ok.assem_t,vecdata_ok))
    matdata_ok = collect_cell_matrix(trial_ok(tn),test,feop_ok.jacs[1](tn,xh_ok,du,dv),trian)
    push!(jac_ok,assemble_matrix(feop_ok.assem_t,matdata_ok))
    test_ptarray(b,res_ok;n)
    test_ptarray(A,jac_ok;n)
  end

  sols_ok,res_ok,jac_ok
end

sols,A,b = pt_quantities(N,Ω)
sols_ok,res_ok,jac_ok = gridap_quantities_for_param(N,Ω)
for i in eachindex(results_ok)
  @assert isapprox(sols.snaps[i][N],sols_ok[i])
  @assert isapprox(sols[N][i],sols_ok[i])
  @assert isapprox(sols.res[N][i],res_ok[i])
  @assert isapprox(sols.jac[N][i],jac_ok[i])
end

# MODE2
nzm = NnzArray(sols)
m2 = change_mode(nzm)
m2_ok = hcat(sols_ok...)'
space_ndofs = size(m2_ok,2)
@assert isapprox(m2_ok,m2[:,(N-1)*space_ndofs+1:N*space_ndofs])
