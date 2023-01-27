include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function poisson_unsteady()
  run_fem = false

  steady = false
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "model.json"
  bnd_info = Dict("dirichlet" => ["sides","sides_c"],
                  "neumann" => ["circle","triangle","square"])
  order = 1

  t0,tF,dt,θ = 0.,5,0.05,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,10.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  op = ParamTransientAffineFEOperator(mfe,lhs,rhs,PS,U,V)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=30,load_offline=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,param_on_structures = offline_phase(info,(uh,μ),(opA,opM,opF,opH),measures,tt)
  online_phase(info,(uh,μ),rbspace,param_on_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  uh,μ = fesol
  uh_offline = uh[1:info.nsnap]
  opA,opM,opF,opH = op

  rbspace = rb(info,tt,uh_offline)

  rbopA = RBVariable(opA,rbspace,rbspace)
  rbopM = RBVariable(opM,rbspace,rbspace)
  rbopF = RBVariable(opF,rbspace)
  rbopH = RBVariable(opH,rbspace)

  Arb = RBOfflineStructure(info,tt,rbopA,μ,meas,:dΩ)
  Mrb = RBOfflineStructure(info,tt,rbopM,μ,meas,:dΩ)
  Frb = RBOfflineStructure(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBOfflineStructure(info,tt,rbopH,μ,meas,:dΓn)

  Arb_eval = eval_off_structure(Arb)
  Mrb_eval = eval_off_structure(Mrb)
  Frb_eval = eval_off_structure(Frb)
  Hrb_eval = eval_off_structure(Hrb)

  Aon_param = RBParamOnlineStructure(Arb,Arb_eval;st_mdeim=info.st_mdeim)
  Mon_param = RBParamOnlineStructure(Mrb,Mrb_eval;st_mdeim=info.st_mdeim)
  Fon_param = RBParamOnlineStructure(Frb,Frb_eval;st_mdeim=info.st_mdeim)
  Hon_param = RBParamOnlineStructure(Hrb,Hrb_eval;st_mdeim=info.st_mdeim)

  param_on_structures = Aon_param,Mon_param,Fon_param,Hon_param
  rbspace,param_on_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::RBSpace,
  param_on_structures::Tuple,
  tt::TimeTracker)

  uh,μ = fesol

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs,rhs = unsteady_poisson_rb_system(expand(param_on_structures),μ[k])
      rb_sol = solve_rb_system(lhs,rhs)
    end
    uhk = get_snap(uh[k])
    uhk_rb = reconstruct_fe_sol(rbspace,rb_sol)
    ErrorTracker(:u,uhk,uhk_rb,k)
  end

  ets = online_loop.(info.online_snaps)
  res = RBResults(:u,tt,ets)
  save(info,res)

  if info.postprocess
    postprocess(info,res)
  end
end

poisson_unsteady()
