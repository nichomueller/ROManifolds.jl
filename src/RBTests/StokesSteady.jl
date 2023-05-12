root = pwd()

@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

function stokes_steady()
  run_fem = false

  steady = true
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  test_path = "$root/tests/stokes/$mesh"
  mesh = "cylinder.json"
  bnd_info = Dict("dirichlet" => ["wall","inlet"],"neumann" => ["outlet"])
  order = 2

  ranges = fill([1.,5.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,b,bfe,f,ffe,h,hfe,g,lhs,rhs = stokes_operators(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTrialFESpace(V,g)
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)
  Y = ParamMultiFieldFESpace([V,Q])
  X = ParamMultiFieldFESpace([U,P])

  op = ParamAffineFEOperator(lhs,rhs,PS,X,Y)

  solver = LinearFESolver()
  nsnap = 100
  uh,ph,μ, = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamOperator(a,afe,PS,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,U,Q;id=:B)
  opF = AffineParamOperator(f,ffe,PS,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,test_path;ϵ=1e-4,nsnap=80,mdeim_snap=30)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rb_space,online_structures = offline_phase(info,(uh,ph,μ),(opA,opB,opF,opH),measures,tt)
  online_phase(info,(uh,ph,μ),rb_space,online_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh,ph,μ = fesol
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  opA,opB,opF,opH = op

  rbspace_u,rbspace_p = assemble_rb_space(info,tt,(uh_offline,ph_offline),opB,ph,μ)

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  Arb = RBAffineDecomposition(info,tt,rbopA,μ,meas,:dΩ)
  Brb = RBAffineDecomposition(info,tt,rbopB,μ,meas,:dΩ)
  Frb = RBAffineDecomposition(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBAffineDecomposition(info,tt,rbopH,μ,meas,:dΓn)

  ad = (Arb,Brb,Frb,Hrb)
  ad_eval = eval_affine_decomposition(ad)
  online_structures = RBParamOnlineStructure(ad,ad_eval)

  rb_space,online_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rb_space::NTuple{2,RBSpace},
  online_structures::Tuple,
  tt::TimeTracker)

  printstyled("Online phase, reduced basis method\n";color=:red)

  uh,ph,μ = fesol

  function online_loop(k::Int)
    printstyled("-------------------------------------------------------------\n")
    printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
    tt.online_time += @elapsed begin
      lhs,rhs = steady_stokes_rb_system(online_structures,μ[k])
      rb_sol = solve_rb_system(lhs,rhs)
    end
    uhk = get_snap(uh[k])
    phk = get_snap(ph[k])
    uhk_rb,phk_rb = reconstruct_fe_sol(rb_space,rb_sol)
    ErrorTracker(:u,uhk,uhk_rb),ErrorTracker(:p,phk,phk_rb)
  end

  ets = online_loop.(info.online_snaps)
  ets_u,ets_p = first.(ets),last.(ets)
  res_u,res_p = RBResults(:u,tt,ets_u),RBResults(:p,tt,ets_p)
  save(info,res_u)
  save(info,res_p)
  printstyled("Average online wall time: $(tt.online_time/length(ets_u)) s";
    color=:red)

  if info.postprocess
    postprocess(info,(res_u,res_p))
  end
end

stokes_steady()
