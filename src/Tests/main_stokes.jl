begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  # mesh = "model_circle_2D_coarse.json"
  mesh = "cube2x2.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  # bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
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

  f(x,μ,t) = VectorValue(0,0)
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)
  # g0(x,μ,t) = VectorValue(0,0)
  # g0(μ,t) = x->g0(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  m(μ,t,(ut,pt),(v,q)) = ∫ₚ(v⋅ut,dΩ)
  lhs(μ,t,(u,p),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  rhs(μ,t,(u,p),(du,dp),(v,q)) = ∫(v*fμt(μ,t),dΩ)

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  # trial_u = PTTrialFESpace(test_u,[g0,g])
  trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTAffineFEOperator(m,lhs,rhs,pspace,trial,test)
  t0,tF,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))
  fesolver = ThetaMethod(LUSolver(),dt,θ)
end

begin
  op,solver = feop,fesolver
  μ = realization(op,2)
  t = dt
  nfree = num_free_dofs(test)
  u = PTArray([zeros(nfree) for _ = 1:2])
  vθ = similar(u)
  vθ .= 0.
  ode_op = get_algebraic_operator(op)
  ode_cache = allocate_cache(ode_op,μ)
  ode_cache = update_cache!(ode_cache,ode_op,μ,t)
  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],vθ)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)

  A = allocate_jacobian(op,uh,ode_cache)
  _matdata_jacobians = fill_jacobians(op,μ,t,xh,(1.,1/t))
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)

  v = get_fe_basis(test)
  b = allocate_residual(op,uh,ode_cache)
  vecdata = collect_cell_vector(test,op.res(μ,t,xh,v))
  assemble_vector_add!(b,op.assem,vecdata)
end

u,v = get_trial_fe_basis(trial_u(nothing,nothing)),get_fe_basis(test_u)
# ptintegrate(aμt(μ,t)*∇(v)⊙∇(u),dΩ.quad)
cf = aμt(μ,t)*∇(v)⊙∇(u)
quad = dΩ.quad
b = change_domain(cf,quad.trian,quad.data_domain_style)
