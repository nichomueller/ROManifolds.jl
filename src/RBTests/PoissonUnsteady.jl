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
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1

  t0,tF,dt,θ = 0.,2.5,0.05,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,20.],6)
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

  opA = NonaffineParamVarOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamVarOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamVarOperator(f,ffe,PS,time_info,V;id=:F)
  opH = NonaffineParamVarOperator(h,hfe,PS,time_info,V;id=:H)

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false,
    save_offline=false,st_mdeim=false)
  tt = TimeTracker(0.,0.)
  rbspace,varinfo = offline_phase(info,(uh,μ),(opA,opM,opF,opH),measures,tt)
  online_phase(info,(uh,μ),rbspace,varinfo,tt)
end

function offline_phase(
  info::RBInfo,
  fe_sol,
  op::NTuple{N,ParamVarOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  uh,μ = fe_sol
  uh_offline = uh[1:info.nsnap]
  opA,opM,opF,opH = op

  rbspace = rb(info,tt,uh_offline)

  rbopA = RBVarOperator(opA,rbspace,rbspace)
  rbopM = RBVarOperator(opM,rbspace,rbspace)
  rbopF = RBVarOperator(opF,rbspace)
  rbopH = RBVarOperator(opH,rbspace)

  A_rb = rb_structure(info,tt,rbopA,μ,meas,:dΩ)
  M_rb = rb_structure(info,tt,rbopM,μ,meas,:dΩ)
  F_rb = rb_structure(info,tt,rbopF,μ,meas,:dΩ)
  H_rb = rb_structure(info,tt,rbopH,μ,meas,:dΓn)

  varinfo = ((rbopA,A_rb),(rbopM,M_rb),(rbopF,F_rb),(rbopH,H_rb))
  rbspace,varinfo
end

function online_phase(
  info::RBInfo,
  fe_sol,
  rbspace::RBSpace,
  varinfo::Tuple,
  tt::TimeTracker)

  uh,μ = fe_sol

  Ainfo,Minfo,Finfo,Hinfo = varinfo
  rbopA,A_rb = Ainfo
  rbopM,M_rb = Minfo
  rbopF,F_rb = Finfo
  rbopH,H_rb = Hinfo

  st_mdeim = info.st_mdeim
  θ = get_θ(rbopA)

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      Aon = online_assembler(rbopA,A_rb,μ[k],st_mdeim)
      Mon = online_assembler(rbopM,M_rb,μ[k],st_mdeim)
      Fon = online_assembler(rbopF,F_rb,μ[k],st_mdeim)
      Hon = online_assembler(rbopH,H_rb,μ[k],st_mdeim)
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
