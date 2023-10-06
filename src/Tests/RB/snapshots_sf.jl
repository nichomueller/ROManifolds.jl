begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/poisson/unsteady/_$mesh"
  order = 1
  degree = 2

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,2.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)
  hμt(μ,t) = PTFunction(h,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)

  jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)
  jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)

  reffe = ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = PTAffineFEOperator(res,jac,jac_t,pspace,trial,test)

  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
end

# INDEX OF TEST
N = 10
μ = realization(feop,N)
times = get_times(fesolver)
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))

function pt_quantities(N,trian)
  # SOL
  sols = collect_solutions(fesolver,feop,μ)

  # RES/JAC
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

  b = allocate_residual(ode_op,_μ,times,_snapsθ,ode_cache)
  vecdata = collect_cell_vector(test,integrate(feop.res(_μ,times,xh,dv)))#,trian)
  assemble_vector_add!(b,feop.assem,vecdata)
  A = allocate_jacobian(ode_op,_μ,times,_snapsθ,ode_cache)
  matdata = collect_cell_matrix(trial(_μ,times),test,integrate(feop.jacs[1](_μ,times,xh,du,dv)))#,trian)
  assemble_matrix_add!(A,feop.assem,matdata)

  sols,b,vecdata,A,matdata
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
  ode_cache_ok = allocate_cache(ode_op_ok)
  w0 = get_free_dof_values(uh0μ(p))
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w0,t0,tf)

  sols_ok = []
  for (uh,t) in sol_gridap
    push!(sols_ok,copy(uh))
  end

  res_ok,jac_ok,vecdata_ok,matdata_ok = [],[],[],[]
  for (n,tn) in enumerate(times)
    xhF_ok = copy(sols_ok[n]),0. * sols_ok[n]
    Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,tn)
    Xh_ok,_,_ = ode_cache_ok
    dxh_ok = ()
    for i in 2:2
      dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],xhF_ok[i]))
    end
    xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],xhF_ok[1]),dxh_ok)
    _vecdata_ok = collect_cell_vector(test,feop_ok.res(tn,xh_ok,dv))#,trian)
    push!(vecdata_ok,_vecdata_ok)
    push!(res_ok,assemble_vector(feop_ok.assem_t,_vecdata_ok))
    _matdata_ok = collect_cell_matrix(trial_ok(tn),test,feop_ok.jacs[1](tn,xh_ok,du,dv))#,trian)
    push!(matdata_ok,_matdata_ok)
    push!(jac_ok,assemble_matrix(feop_ok.assem_t,_matdata_ok))
  end

  sols_ok,res_ok,vecdata_ok,jac_ok,matdata_ok
end

sols,b,vecdata,A,matdata = pt_quantities(N,Ω)
sols_ok,res_ok,vecdata_ok,jac_ok,matdata_ok = gridap_quantities_for_param(N,Ω)
ntimes = length(times)
for i in eachindex(sols_ok)
  @assert isapprox(sols.snaps[i][N],sols_ok[i])
  @assert isapprox(sols[N][i],sols_ok[i])
  #@assert isapprox(b[(N-1)*ntimes+i],res_ok[i]) "not true for index $i"
  @assert isapprox(vecdata[1][1][(N-1)*ntimes+i],vecdata_ok[1][1][i]) "not true for index $i"
  @assert isapprox(A[(N-1)*ntimes+i],jac_ok[i])
  @assert isapprox(matdata[1][1][(N-1)*ntimes+i],matdata_ok[i][1][1])
end

# # MODE2
# nzm = NnzArray(sols)
# m2 = change_mode(nzm)
# m2_ok = hcat(sols_ok...)'
# space_ndofs = size(m2_ok,2)
# @assert isapprox(m2_ok,m2[:,(N-1)*space_ndofs+1:N*space_ndofs])

# INVESTIGATE JACOBIAN
solsθ = recenter(fesolver,sols,μ)[1:N]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ[1:N],times)
ode_cache = update_cache!(ode_cache,ode_op,μ[1:N],times)
Xh, = ode_cache
dxh = ()
_xh = (solsθ,solsθ-solsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

# DOMAIN CONTRIB
dc = integrate(feop.jacs[1](μ[1:N],times,xh,du,dv))
dc_ok = [∫(a(p,t)*∇(dv)⋅∇(du))dΩ for p in μ for t in times]
for n in eachindex(dc_ok)
  test_ptarray(dc[Ω],dc_ok[n][Ω];n)
end

# CELL DATA
g_ok(x,t) = g(x,rand(3),t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
matdata = collect_cell_matrix(trial(μ,times),test,dc)
ntimes = length(times)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  test_ptarray(matdata[1][1],matdata_ok[1][1];n)
end

# ALGEBRAIC STRUCTURE
A = allocate_jacobian(ode_op,μ,times,solsθ,ode_cache)
A0 = copy(A[1])
assemble_matrix_add!(A,feop.assem,matdata)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  A_ok = copy(A0)
  assemble_matrix_add!(A_ok,feop.assem,matdata_ok)
  test_ptarray(A,A_ok;n)
end

# INVESTIGATE RESIDUAL
solsθ = recenter(fesolver,sols,μ)[N:N]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ[N:N],times)
ode_cache = update_cache!(ode_cache,ode_op,μ[N:N],times)
Xh, = ode_cache
dxh = ()
_xh = (solsθ,solsθ-solsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

# DOMAIN CONTRIB
res_fun(p,t,u,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) #- ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn) #∫ₚ(v*∂ₚt(u),dΩ) +
dc = integrate(res_fun(μ[N:N],times,xh,dv))
dc_ok = []
res_ok_fun(p,t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ #- ∫(v*f(p,t))dΩ - ∫(v*h(p,t))dΓn # ∫(∂t(u)*v)dΩ +
for (nt,t) in enumerate(times)
  ode_op_ok = get_algebraic_operator(feop_ok)
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok, = ode_cache_ok
  dxh_ok = ()
  _xh_ok = (sols_ok[nt],sols_ok[nt]-sols_ok[nt])
  for i in 2:get_order(feop)+1
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],_xh_ok[i]))
  end
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],_xh_ok[1]),dxh_ok)
  push!(dc_ok,res_ok_fun(μ[N],t,xh_ok,dv))
end
for n in eachindex(dc_ok)
  test_ptarray(dc[Ω],dc_ok[n][Ω];n)
end

dc1 = collect(dc[Ω][1])
dc_ok1 = collect(dc_ok[1][Ω])


# CELL DATA
g_ok(x,t) = g(x,rand(3),t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
matdata = collect_cell_matrix(trial(μ,times),test,dc)
ntimes = length(times)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  test_ptarray(matdata[1][1],matdata_ok[1][1];n)
end

# ALGEBRAIC STRUCTURE
A = allocate_jacobian(ode_op,μ,times,solsθ,ode_cache)
A0 = copy(A[1])
assemble_matrix_add!(A,feop.assem,matdata)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  A_ok = copy(A0)
  assemble_matrix_add!(A_ok,feop.assem,matdata_ok)
  test_ptarray(A,A_ok;n)
end










# @check all([A[n] == assemble_matrix(∫(a(params[slow_idx(n,ntimes)],times[fast_idx(n,ntimes)])*∇(dv)⋅∇(du))dΩ,
#   trial_ok(times[fast_idx(n,ntimes)]),test) for n = 1:100])
nparams = length(μ)
@check all([A[n] == assemble_matrix(∫(a(μ[fast_idx(n,nparams)],times[slow_idx(n,nparams)])*∇(dv)⋅∇(du))dΩ,
  trial_ok(times[slow_idx(n,nparams)]),test) for n = 1:100])

n = 1
for p in μ, t in times
  Aok = assemble_matrix(∫(a(p,t)*∇(dv)⋅∇(du))dΩ,trial_ok(t),test)
  A_ok = copy(A0)
  matdata_ok = collect_cell_matrix(trial_ok(t),test,∫(a(p,t)*∇(dv)⋅∇(du))dΩ)
  assemble_matrix_add!(A_ok,feop.assem,matdata_ok)
  @check Aok == A_ok
  @check A[n] == Aok "not true for n = $n"
  n += 1
end

_μ = μ[1:2]
solsθ = recenter(fesolver,sols,μ)[1:2]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,_μ,times)
ode_cache = update_cache!(ode_cache,ode_op,_μ,times)
Xh, = ode_cache
dxh = ()
_xh = (solsθ,solsθ-solsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
A = allocate_jacobian(ode_op,μ,times,solsθ,ode_cache)
assemble_matrix_add!(A,feop.assem,matdata)

for (n,t) in enumerate(times)
  @check assemble_matrix(∫(a(_μ[1],t)*∇(dv)⋅∇(du))dΩ,trial_ok(t),test) == A[n]
end

for (n,t) in enumerate(times)
  @check assemble_matrix(∫(a(_μ[2],t)*∇(dv)⋅∇(du))dΩ,trial_ok(t),test) == A[ntimes+n]
end
