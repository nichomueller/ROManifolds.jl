begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
  mesh = "cube2x2.json"
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  order = 2
  degree = 4

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  m(μ,t,u,v) = ∫ₚ(v⋅u,dΩ)
  a(μ,t,(u,p),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  c(μ,t,u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
  dc(μ,t,u,du,v) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)

  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = m(μ,t,dut,v)
  jac(μ,t,(u,p),(du,dp),(v,q)) = a(μ,t,(du,dp),(v,q)) + dc(μ,t,u,du,v)
  res(μ,t,(u,p),(v,q)) = m(μ,t,∂ₚt(u),v) + a(μ,t,(u,p),(v,q)) + c(μ,t,u,v)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,0.5
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

  nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
  fesolver = PThetaMethod(nls,xh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = false
  save_solutions = true
  load_structures = false
  save_structures = true
  energy_norm = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_system = 20
  nsnaps_test = 10
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_solutions,save_solutions,load_structures,save_structures,
                energy_norm,compute_supremizers,nsnaps_state,nsnaps_system,nsnaps_test,st_mdeim)
  # multi_field_rb_model(info,feop,fesolver)
end

nsnaps = info.nsnaps_state
params = realization(feop,nsnaps)
trial = get_trial(feop)
sols,stats = collect_solutions(fesolver,feop,trial,params)
save(info,(sols,params,stats))

rbspace = reduced_basis(info,feop,sols,params)
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
save(info,(rbspace,rbrhs,rblhs))

snaps_test,params_test = load_test(info,feop,fesolver)

println("Solving nonlinear RB problems with Newton iterations")
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
nl_cache = nothing
x = initial_guess(sols,params,params_test)
xrb = space_time_projection(x,rbspace)
_,conv0 = Algebra._check_convergence(fesolver.nls.ls,xrb)
iter = 1
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,x,params_test)
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,params_test)
nl_cache = rb_solve!(xrb,fesolver.nls.ls,rhs,lhs,nl_cache)
x .= recast(rbspace,xrb)
isconv,conv = Algebra._check_convergence(fesolver.nls,xrb,conv0)
println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
if all(isconv); return; end
if iter == nls.max_nliters
  @unreachable
end
post_process(info,feop,fesolver,snaps_test,params_test,x,stats)

μ = params
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,dt)
w0 = get_free_dof_values(xh0μ(μ))
vθ = similar(w0)
vθ .= 0.0
nl_cache = nothing
Us,_,fecache = update_cache!(ode_cache,ode_op,μ,dt)
uh = EvaluationFunction(Us[1],vθ)
n = length(uh)
μ1 = isa(μ,Table) ? testitem(μ) : μ
t1 = isa(dt,AbstractVector) ? testitem(dt) : dt
uh1 = testitem(uh)
V = get_test(feop)
v = get_fe_basis(V)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh1)
end
xh = TransientCellField(uh1,dxh)
dc = integrate(feop.res(μ1,t1,xh,v))

mm = m(μ1,t1,∂ₚt(xh[1]),v[1])
aa = a(μ1,t1,xh,v)
cc = c(μ1,t1,xh[1],v[1])

contrib = CollectionPTIntegrand()
add_contribution!(contrib,+,mm)
for (_op,int) in aa.dict
  add_contribution!(contrib,+,int...)
end

du1 = get_trial_fe_basis(trial_u(nothing,nothing))
du = [du1,nothing]
dv1 = get_fe_basis(test_u)
dv = [dv1,nothing]
μ = params
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,dt)
w0 = get_free_dof_values(xh0μ(μ))
vθ = similar(w0)
vθ .= 0.0
nl_cache = nothing
Us,_,fecache = update_cache!(ode_cache,ode_op,μ,dt)
Xh, = ode_cache
dxh = ()
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],vθ))
end
xh = TransientCellField(EvaluationFunction(Xh[1],vθ),dxh)
xh = [xh[1],nothing]
∫ₚ(dv1⊙(dconv∘(du1,∇(du1),xh[1],∇(xh[1]))),dΩ)
nt = nothing
∫ₚ(nt⊙(dconv∘(nt,∇(nt),nt,∇(nt))),dΩ)

feop_row_col = feop[1,2]
u = zero(feop_row_col.test)
j(du,dv) = integrate(feop_row_col.jacs[1](μ[1],dt,u,du,dv))
trial_dual = get_trial(feop_row_col)
# assemble_matrix(j,trial_dual(μ[1],dt),feop_row_col.test)
integrate(feop_row_col.jacs[1](μ[1],dt,zero(test_u),
  get_trial_fe_basis(trial_p),get_fe_basis(test_u)))

∫ₚ(get_fe_basis(test_u)⊙(dconv∘(nt,∇(nt),nt,nt)),dΩ)

x1 = x[1]
rb1 = rbspace[1]
time_ndofs = get_time_ndofs(rb1)
ptproj = map(1:time_ndofs) do n
  mat = hcat(a[(n-1)*time_ndofs+1:n*time_ndofs]...)
  space_time_projection(mat,rb1)
end
PTArray(ptproj)
