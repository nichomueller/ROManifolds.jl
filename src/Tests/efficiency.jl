# check efficiency
wf,w0,t0,solver,cache = copy(w),copy(w),dt,fesolver,nothing
@time begin
  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ
  ode_cache = allocate_cache(ode_op,μ)
  vθ = similar(w0)
  vθ .= 0.0
  l_cache = nothing
  A,b = _allocate_matrix_and_vector(ode_op,w0,ode_cache)
  ode_cache = update_cache!(ode_cache,ode_op,μ,tθ)
  _matrix_and_vector!(A,b,ode_op,μ,tθ,dtθ,w0,ode_cache,vθ)
  afop = PAffineOperator(A,b)
  l_cache = solve!(wf,solver.nls,afop,l_cache)
  @. wf.array = wf.array + w0.array
end

wf_ok,w0_ok,cache_ok = copy(w[1]),copy(w[1]),nothing
@time begin
  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  vθ_ok = similar(w0_ok)
  vθ_ok .= 0.0
  l_cache_ok = nothing
  Aok,bok = Gridap.ODEs.ODETools._allocate_matrix_and_vector(ode_op_ok,w0_ok,ode_cache_ok)
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,tθ)
  Gridap.ODEs.ODETools._matrix_and_vector!(Aok,bok,ode_op_ok,tθ,dtθ,w0_ok,ode_cache_ok,vθ_ok)
  afop_ok = AffineOperator(Aok,bok)
  l_cache_ok = solve!(wf_ok,solver.nls,afop_ok,l_cache_ok)
  @. wf_ok = wf_ok + w0_ok
end

# investigate efficiency of _matrix_and_vector!
@time _matrix!(A,ode_op,μ,tθ,dtθ,w0,ode_cache,vθ)
@time _vector!(b,ode_op,μ,tθ,dtθ,w0,ode_cache,vθ)
@time begin
  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],(w0,vθ)[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],(w0,vθ)[1]),dxh)
end
@time begin
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(μ,dt,xh,v))
end

@time Gridap.ODEs.ODETools._matrix!(Aok,ode_op_ok,tθ,dtθ,w0_ok,ode_cache_ok,vθ_ok)
@time Gridap.ODEs.ODETools._vector!(bok,ode_op_ok,tθ,dtθ,w0_ok,ode_cache_ok,vθ_ok)
@time begin
  Xh_ok, = ode_cache_ok
  dxh_ok = ()
  for i in 2:get_order(op)+1
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],(w0_ok,vθ_ok)[i]))
  end
  xh_ok=TransientCellField(EvaluationFunction(Xh_ok[1],(w0_ok,vθ_ok)[1]),dxh_ok)
end
@time begin
  V_ok = feop_ok.test
  v_ok = get_fe_basis(V_ok)
  vecdata = collect_cell_vector(V_ok,feop_ok.res(dt,xh_ok,v_ok))
end

@time evaluate(∫ₚ(v*∂ₚt(xh) + aμt(μ,dt)*∇(v)⋅∇(xh) - fμt(μ,dt)*v,dΩ))
@time evaluate(∫ₚ(hμt(μ,dt)*v,dΓn))
@time ∫(v*∂t(xh_ok))dΩ + ∫(a(μ[1],dt)*∇(v)⋅∇(xh_ok))dΩ - ∫(f(μ[1],dt)*v)dΩ
@time ∫(h(μ[1],dt)*v)dΓn
