include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function poisson_unsteady()
  run_fem = true

  steady = false
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "model.json"
  bnd_info = Dict("dirichlet" => ["sides","sides_c"],
                  "neumann" => ["circle","triangle","square"])
  order = 1

  t0,tF,dt,θ = 0.,2.5,0.05,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,3.],6)
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

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false,
    st_mdeim=true,fun_mdeim=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,rb_structures = offline_phase(info,(uh,μ),(opA,opM,opF,opH),measures,tt)
  online_phase(info,(uh,μ),rbspace,rb_structures,tt)
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

  Arb = RBStructure(info,tt,rbopA,μ,meas,:dΩ)
  Mrb = RBStructure(info,tt,rbopM,μ,meas,:dΩ)
  Frb = RBStructure(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBStructure(info,tt,rbopH,μ,meas,:dΓn)

  rb_structures = ((rbopA,Arb),(rbopM,Mrb),(rbopF,Frb),(rbopH,Hrb))
  rbspace,rb_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::RBSpace,
  rb_structures::Tuple,
  tt::TimeTracker)

  uh,μ = fesol

  Arb,Mrb,Frb,Hrb = rb_structures
  rbopA,Arb = Arb
  rbopM,Mrb = Mrb
  rbopF,Frb = Frb
  rbopH,Hrb = Hrb

  st_mdeim = info.st_mdeim
  θ = get_θ(rbopA)

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      Aon = online_assembler(rbopA,Arb,μ[k],st_mdeim)
      Mon = online_assembler(rbopM,Mrb,μ[k],st_mdeim)
      Fon = online_assembler(rbopF,Frb,μ[k],st_mdeim)
      Hon = online_assembler(rbopH,Hrb,μ[k],st_mdeim)
      lift = Aon[2],Mon[2]
      sys = poisson_rb_system((Aon[1]...,Mon[1]...),(Fon,Hon,lift...),θ)
      rb_sol = solve_rb_system(sys...)
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
