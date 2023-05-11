root = pwd()

@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

function navier_stokes_steady()
  run_fem = false

  steady = true
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  test_path = "$root/tests/navier-stokes/$mesh"
  mesh = "cylinder.json"
  bnd_info = Dict("dirichlet" => ["wall","inlet"],"neumann" => ["outlet"])
  order = 2

  ranges = fill([1.,2.],9)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,b,bfe,c,cfe,d,dfe,f,ffe,h,hfe,g,res,jac = navier_stokes_functions(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTrialFESpace(V,g)
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)
  Y = ParamMultiFieldFESpace([V,Q])
  X = ParamMultiFieldFESpace([U,P])

  op = ParamFEOperator(res,jac,PS,X,Y)

  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = FESolver(nls)
  nsnap = 100
  uh,ph,μ, = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamOperator(a,afe,PS,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,U,Q;id=:B)
  opC = NonlinearParamOperator(c,cfe,PS,U,V;id=:C)
  opD = NonlinearParamOperator(d,dfe,PS,U,V;id=:D)
  opF = AffineParamOperator(f,ffe,PS,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,test_path;ϵ=1e-4,nsnap=80,mdeim_snap=30,load_offline=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  fesol = (uh,ph,μ,U,V)
  rbspace,online_structures = offline_phase(info,fesol,(opA,opB,opC,opD,opF,opH),measures,tt)
  online_phase(info,fesol,rbspace,online_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh,ph,μ, = fesol
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  opA,opB,opC,opD,opF,opH = op

  rbspace_u,rbspace_p = assemble_rbspace(info,tt,(uh_offline,ph_offline),opB,ph,μ)
  rbspace = rbspace_u,rbspace_p

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopC = RBVariable(opC,rbspace_u,rbspace_u)
  rbopD = RBVariable(opD,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  Arb = RBAffineDecomposition(info,tt,rbopA,μ,meas,:dΩ)
  Brb = RBAffineDecomposition(info,tt,rbopB,μ,meas,:dΩ)
  Crb = RBAffineDecomposition(info,tt,rbopC,μ,meas,:dΩ)
  Drb = RBAffineDecomposition(info,tt,rbopD,μ,meas,:dΩ)
  Frb = RBAffineDecomposition(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBAffineDecomposition(info,tt,rbopH,μ,meas,:dΓn)

  ad = (Arb,Brb,BTrb,Crb,Drb,Mrb,Frb,Hrb)
  ad_eval = eval_affine_decomposition(ad)
  online_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

  rbspace,online_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::NTuple{2,RBSpace},
  online_structures::Tuple,
  tt::TimeTracker)

  printstyled("Online phase, reduced basis method\n";color=:red)

  uh,ph,μ,U = fesol
  μ_offline = μ[1:info.nsnap]
  rb_solver(res,jac,x0,Uk) = solve_rb_system(res,jac,x0,Uk,rbspace)

  function online_loop(k::Int)
    printstyled("-------------------------------------------------------------\n")
    printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
    tt.online_time += @elapsed begin
      res,jac = unsteady_navier_stokes_rb_system(online_structures,μ[k])
      Uk = get_trial(U)(μ[k])
      x0 = get_initial_guess(uh,ph,μ_offline,μ[k])
      rb_sol = rb_solver(res,jac,x0,Uk)
    end
    uhk,phk = get_snap(uh[k]),get_snap(ph[k])
    uhk_rb,phk_rb = reconstruct_fe_sol(rbspace,rb_sol)
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

navier_stokes_steady()
