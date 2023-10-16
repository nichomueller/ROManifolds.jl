begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
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

  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  res(μ,t,(u,p),(v,q)) = (∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ)
    - ∫ₚ(q*(∇⋅(u)),dΩ) - ∫ₚ(v⋅fμt(μ,t),dΩ))

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  # trial_u = PTTrialFESpace(test_u,[g0,g])
  trial_u = PTTrialFESpace(test_u,g)
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
  load_solutions = true
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

# lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,params_test)
T = Float
njacs = length(rblhs)
nblocks = get_nblocks(testitem(rblhs))
offsets = field_offsets(feop.test)
rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
i = 1
rb_jac_i = rblhs[i]
rb_offsets_row = field_offsets(rb_jac_i)
blocks = Matrix{PTArray{Matrix{T}}}(undef,nblocks,nblocks)
for (row,col) = index_pairs(nblocks,nblocks)
  cache_row_col = cache_at_index(lhs_cache,offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1])
  if rb_jac_i.touched[row,col]
    feop_row_col = feop[row,col]
    sols_col = x[col]
    blocks[row,col] = collect_lhs_contributions!(
      cache_row_col,info,feop_row_col,fesolver,rb_jac_i.blocks[row,col],sols_col,params_test;i)
  else
    rbcache,_ = last(cache_row_col)
    s = (rb_offsets_i[row+1]-rb_offsets_i[row],rb_offsets_i[col+1]-rb_offsets_i[col])
    setsize!(rbcache,s)
    array = rbcache.array
    array .= zero(T)
    blocks[row,col] = PTArray([copy(array) for _ = eachindex(params_test)])
  end
end
rb_jacs_contribs[i] = hvcat(nblocks,blocks...)

nrows = Int(length(blocks)/nblocks)
harray = map(1:nrows) do row
  hcat(blocks[(row-1)*nblocks:row*nblocks]...)
end
hvarray = vcat(harray...)
hvarray
