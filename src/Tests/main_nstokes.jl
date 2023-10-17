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
