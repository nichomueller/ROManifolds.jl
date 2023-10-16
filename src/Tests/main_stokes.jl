begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
  mesh = "model_circle_2D_coarse.json"
  # mesh = "cube2x2.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  # bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
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

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)
  g0(x,μ,t) = VectorValue(0,0)
  g0(μ,t) = x->g0(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  res(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial_u = PTTrialFESpace(test_u,[g0,g])
  # trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),xh0μ,θ,dt,t0,tf)

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
  multi_field_rb_model(info,feop,fesolver)
end

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
rbrhs,rblhs = load(info,(BlockRBVecAlgebraicContribution,Vector{BlockRBMatAlgebraicContribution}))

nsnaps = info.nsnaps_state
params = realization(feop,nsnaps)
sols,stats = collect_solutions(fesolver,feop,trial,params)
save(info,(sols,params,stats))
rbspace = reduced_basis(info,feop,sols,params)
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
save(info,(rbspace,rbrhs,rblhs))

snaps_test,params_test = load_test(info,feop,fesolver)
x = initial_guess(sols,params,params_test)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,x,params_test)
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,params_test)
stats = @timed begin
  rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs)
end
approx_snaps_test = recast(rbspace,rb_snaps_test)
post_process(info,feop,fesolver,snaps_test,params_test,approx_snaps_test,stats)

μ = params_test[1]
bsu,btu = rbspace[1].basis_space,rbspace[1].basis_time
bsp,btp = rbspace[2].basis_space,rbspace[2].basis_time

urb = space_time_projection(hcat(snaps_test[1][1:10]...),rbspace[1])
prb = space_time_projection(hcat(snaps_test[2][1:10]...),rbspace[2])
_urb = rb_snaps_test[1][1:get_rb_ndofs(rbspace[1])]
_prb = rb_snaps_test[1][1+get_rb_ndofs(rbspace[1]):end]

Aμ(t) = assemble_matrix((du,dv)->∫(a(μ,t)*∇(dv)⊙∇(du))dΩ,trial_u(μ,t),test_u)
Mμ(t) = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μ,dt),test_u)/(dt*θ)
A = NnzMatrix([NnzVector(Aμ(t)) for t in get_times(fesolver)]...)
M = NnzMatrix([NnzVector(Mμ(t)) for t in get_times(fesolver)]...)
Arb = space_time_projection(A,rbspace[1],rbspace[1];combine_projections=(x,y)->θ*x+(1-θ)*y)
Mrb = space_time_projection(M,rbspace[1],rbspace[1];combine_projections=(x,y)->θ*x-θ*y)
lhs1_ok = Arb+Mrb
lhs1 = lhs[1][1:get_rb_ndofs(rbspace[1]),1:get_rb_ndofs(rbspace[1])]
println(ℓ∞(lhs1_ok - lhs1))

BT1 = -assemble_matrix((dp,dv)->∫(dp*(∇⋅(dv)))dΩ,trial_p,test_u)
BTμ = [BT1 for _ = 1:get_time_ndofs(fesolver)]
BT = NnzMatrix(NnzVector.(BTμ)...)
BTrb = space_time_projection(BT,rbspace[1],rbspace[2];combine_projections=(x,y)->θ*x+(1-θ)*y)
lhs2_ok = BTrb
lhs2 = lhs[1][1:get_rb_ndofs(rbspace[1]),get_rb_ndofs(rbspace[1])+1:end]
println(ℓ∞(lhs2_ok - lhs2))

B1 = -assemble_matrix((du,dq)->∫(dq*(∇⋅(du)))dΩ,trial_u(nothing,nothing),test_p)
Bμ = [B1 for _ = 1:get_time_ndofs(fesolver)]
B = NnzMatrix(NnzVector.(Bμ)...)
Brb = space_time_projection(B,rbspace[2],rbspace[1];combine_projections=(x,y)->θ*x+(1-θ)*y)
lhs3_ok = Brb
lhs3 = lhs[1][get_rb_ndofs(rbspace[1])+1:end,1:get_rb_ndofs(rbspace[1])]
println(ℓ∞(lhs3_ok - lhs3))

dir(t) = zero(trial_u(μ,t))
ddir(t) = zero(∂ₚt(trial_u)(μ,t))
Lu(t) = assemble_vector(dv->∫(a(μ,t)*∇(dv)⊙∇(dir(t)))dΩ,test_u) + assemble_vector(dv->∫(dv⋅ddir(t))dΩ,test_u)
Lp(t) = assemble_vector(dq->∫(dq*(∇⋅(dir(t))))dΩ,test_p)

Lut = NnzMatrix([Lu(t) for t in get_times(fesolver)]...)
Lpt = NnzMatrix([Lp(t) for t in get_times(fesolver)]...)
Lurb = space_time_projection(Lut,rbspace[1])
Lprb = space_time_projection(Lpt,rbspace[2])
rhs1_ok = Lurb
rhs1 = rhs[1][1:get_rb_ndofs(rbspace[1])]
println(ℓ∞(rhs1_ok - rhs1))
rhs2_ok = Lprb
rhs2 = rhs[1][1+get_rb_ndofs(rbspace[1]):end]
println(ℓ∞(rhs2_ok - rhs2))
