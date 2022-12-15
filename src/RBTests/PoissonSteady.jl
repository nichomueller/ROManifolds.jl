include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function poisson_steady()
  run_fem = false

  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "cube15x15x15.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1

  ranges = fill([1.,20.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamAffineFEOperator(lhs,rhs,PS,U,V)

  solver = LinearFESolver()
  nsnap = 100
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamVarOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = NonaffineParamVarOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false)
  tt = TimeTracker(0.,0.)
  rbspace,varinfo = offline_phase(info,(uh,μ),(opA,opF,opH),measures,tt)
  online_phase(info,(uh,μ),rbspace,varinfo,tt)
end

function offline_phase(
  info::RBInfo,
  fe_sol,
  op::Vector{<:ParamVarOperator},
  meas::ProblemMeasures,
  tt::TimeTracker)

  uh,μ = fe_sol
  uh_offline = uh[1:info.nsnap]
  opA,opF,opH = op

  rbspace = rb(info,tt,uh_offline)
  rbopA = RBVarOperator(opA,rbspace,rbspace)
  rbopF = RBVarOperator(opF,rbspace)
  rbopH = RBVarOperator(opH,rbspace)

  if info.load_offline
    A_rb = load_rb_structure(info,rbopA,meas.dΩ)
    F_rb = load_rb_structure(info,rbopF,meas.dΩ)
    H_rb = load_rb_structure(info,rbopH,meas.dΓn)
  else
    A_rb = assemble_rb_structure(info,tt,rbopA,μ,meas,:dΩ)
    F_rb = assemble_rb_structure(info,tt,rbopF,μ,meas,:dΩ)
    H_rb = assemble_rb_structure(info,tt,rbopH,μ,meas,:dΓn)
  end

  varinfo = ((rbopA,A_rb),(rbopF,F_rb),(rbopH,H_rb))
  rbspace,varinfo
end

function online_phase(
  info::RBInfo,
  fe_sol,
  rbspace::RBSpace,
  varinfo::Tuple,
  tt::TimeTracker)

  uh,μ = fe_sol

  Ainfo,Finfo,Hinfo = varinfo
  rbopA,A_rb = Ainfo
  rbopF,F_rb = Finfo
  rbopH,H_rb = Hinfo

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs = online_assembler(rbopA,A_rb,μ[k])
      rhs = online_assembler(rbopF,F_rb,μ[k]),online_assembler(rbopH,H_rb,μ[k])
      sys = poisson_rb_system(lhs[1],(rhs...,lhs[2]))
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

poisson_steady()
