begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

function heat_equation()
  root = pwd()
  mesh = "elasticity_3cyl2D.json"
  bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  # mesh = "cube2x2.json"
  # bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/poisson/unsteady/$mesh"
  order = 1
  degree = 2
  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,10.],3)
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

  res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
  jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

  T = Float
  reffe = ReferenceFE(lagrangian,T,order)
  feinfo = FEInfo(reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.3,0.005,0.5
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  norm_style = :l2
  nsnaps_state = 50
  nsnaps_mdeim = 20
  nsnaps_test = 10
  st_mdeim = false
  rbinfo = RBInfo(feinfo,test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

  # Offline phase
  printstyled("OFFLINE PHASE\n";bold=true,underline=true)
  if load_solutions
    sols,params = load(rbinfo,(Snapshots{Vector{T}},Table))
  else
    params = realization(feop,nsnaps_state+nsnaps_test)
    sols,stats = collect_single_field_solutions(fesolver,feop,params)
    if save_solutions
      save(rbinfo,(sols,params,stats))
    end
  end
  if load_structures
    rbspace = load(rbinfo,RBSpace{T})
    rbrhs,rblhs = load(rbinfo,(RBVecAlgebraicContribution{T},
      Vector{RBMatAlgebraicContribution{T}}))
  else
    rbspace = reduced_basis(rbinfo,feop,sols)
    rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
    if save_structures
      save(rbinfo,(rbspace,rbrhs,rblhs))
    end
  end
  # Online phase
  printstyled("ONLINE PHASE\n";bold=true,underline=true)
  rb_solver(rbinfo,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
end

heat_equation()


μ = params[1:nsnaps_mdeim]
op = get_ptoperator(fesolver,feop,rbspace,μ)
nzm,trian = collect_residuals_for_trian(op)
_nzm,_trian = nzm[1],Γn
basis_space,basis_time = compress(_nzm;ϵ=rbinfo.ϵ)
proj_bs,proj_bt = project_space_time(basis_space,basis_time,rbspace)
interp_idx_space = get_interpolation_idx(basis_space)
interp_bs = basis_space[interp_idx_space,:]
if rbinfo.st_mdeim
  interp_idx_time = get_interpolation_idx(basis_time)
  interp_bt = basis_time[interp_idx_time,:]
  interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
  lu_interp = lu(interp_bst)
else
  lu_interp = lu(interp_bs)
  interp_idx_time = collect(eachindex(op.tθ))
end
# integr_domain = RBIntegrationDomain(feinfo,op.odeop.feop,nzm,trian,interp_idx_space,interp_idx_time)
recast_idx_space = recast_idx(_nzm,interp_idx_space)
recast_idx_space_rows,_ = vec_to_mat_idx(recast_idx_space,_nzm.nrows)
cell_dof_ids = get_cell_dof_ids(feop.test,_trian)
red_integr_cells = find_cells(recast_idx_space_rows,cell_dof_ids)
model = get_background_model(_trian)
red_model = DiscreteModelPortion(model,red_integr_cells)
red_feop = reduce_feoperator(feop,feinfo,red_model)
red_trian = Triangulation(red_model)
red_meas = Measure(red_trian,2*get_order(feop.test))
