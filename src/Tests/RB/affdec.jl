# collect_compress_rhs(info,feop,fesolver,rbspace,snaps,params)
nsnaps = info.nsnaps_system
times = get_times(fesolver)
θ = fesolver.θ

nsnaps = info.nsnaps_system
snapsθ = recenter(fesolver,snaps,params)
sols,μ = snapsθ[1:nsnaps],params[1:nsnaps]
# ress,meas = collect_residuals(fesolver,feop,sols,μ)
times = get_times(fesolver)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
b = allocate_residual(ode_op,sols,ode_cache)
dt,θ = fesolver.dt,fesolver.θ
dtθ = θ == 0.0 ? dt : dt*θ
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,sols,ode_cache,sols)
separate_contribs = Val(true)
xhF = (sols,sols-sols)
Xh, = ode_cache
dxh = ()
for i in 2:get_order(ode_op)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
end
xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
V = get_test(feop)
v = get_fe_basis(V)
dc = feop.res(μ,times,xh,v)
nmeas = num_domains(dc)
meas = get_domains(dc)
bvec = Vector{typeof(b)}(undef,nmeas)
for (n,m) in enumerate(meas)
  vecdata = collect_cell_vector(V,dc,m)
  assemble_vector_add!(b,feop.assem,vecdata)
  bvec[n] = copy(b)
end

ad_res = compress_component(info,feop,ress,meas,times,rbspace)
