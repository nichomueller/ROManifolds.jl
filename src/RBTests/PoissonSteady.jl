include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function poisson_steady()
  run_fem = true

  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "model.json"
  bnd_info = Dict("dirichlet" => ["sides","sides_c"],
                  "neumann" => ["circle","triangle","square"])
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
  nsnap = 1
  uh,μ, = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamOperator(f,ffe,PS,V;id=:F)
  opH = NonaffineParamOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,rb_structures = offline_phase(info,(uh,μ),(opA,opF,opH),measures,tt)
  online_phase(info,(uh,μ),rbspace,rb_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::Vector{<:ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker)

  uh,μ = fesol
  uh_offline = uh[1:info.nsnap]
  opA,opF,opH = op

  rbspace = rb(info,tt,uh_offline)
  rbopA = RBVariable(opA,rbspace,rbspace)
  rbopF = RBVariable(opF,rbspace)
  rbopH = RBVariable(opH,rbspace)

  if info.load_offline
    Arb = load(info,rbopA,meas.dΩ)
    Frb = load(info,rbopF,meas.dΩ)
    Hrb = load(info,rbopH,meas.dΓn)
  else
    Arb = assemble_rb_structure(info,tt,rbopA,μ,meas,:dΩ)
    Frb = assemble_rb_structure(info,tt,rbopF,μ,meas,:dΩ)
    Hrb = assemble_rb_structure(info,tt,rbopH,μ,meas,:dΓn)
  end

  rb_structures = ((rbopA,Arb),(rbopF,Frb),(rbopH,Hrb))
  rbspace,rb_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::RBSpace,
  rb_structures::Tuple,
  tt::TimeTracker)

  uh,μ = fesol

  Arb,Frb,Hrb = rb_structures
  rbopA,Arb = Arb
  rbopF,Frb = Frb
  rbopH,Hrb = Hrb

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs = online_assembler(rbopA,Arb,μ[k])
      rhs = online_assembler(rbopF,Frb,μ[k]),online_assembler(rbopH,Hrb,μ[k])
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
