include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("tests.jl")

function poisson_steady()
  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)
  run_fem = true

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1
  degree = get_degree(order)

  ranges = fill([1.,10.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  model = model_info(bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamAffineFEOperator(lhs,rhs,PS,get_trial(U),get_test(V))

  solver = LinearFESolver()
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,100)

  opA = ParamVarOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = ParamVarOperator(h,hfe,PS,V;id=:H)

  rbinfo = RBInfo(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_nsnap=20)

  tt,basis,rb_A,rb_F,rb_H = offline_phase(rbinfo,[uh,μ],[opA,opF,opH],measures)
  online_phase(info,[uh,μ],basis,tt,rb_A,rb_F,rb_H)
end

function offline_phase(
  info::RBInfo,
  fe_sol::Vector{Snapshots},
  op::Vector{<:RBVarOperator},
  meas::ProblemMeasures)

  uh,μ = fe_sol
  opA,opF,opH = op
  tt = TimeTracker(0.,0.)

  if info.load_offline
    basis = get_rb(info)
    rb_A = load_rb_structure(info,opA,basis,basis,tt,μ,meas,:dΩ)
    rb_F = load_rb_structure(info,opF)
    rb_H = load_rb_structure(info,opH,basis,tt,μ,meas,:dΓn)
    tt,basis,rb_A,rb_F,rb_H
  else
    basis = assemble_rb(info,tt,uh)
    rbopA = RBVarOperator(opA,basis,basis)
    rbopF = RBVarOperator(opF,basis)
    rbopH = RBVarOperator(opH,basis)
    rb_A = assemble_rb_structures(info,tt,rbopA,μ,meas,:dΩ)
    rb_F = assemble_rb_structures(info,tt,rbopF)
    rb_H = assemble_rb_structures(info,tt,rbopH,μ,meas,:dΓn)
    tt,basis,rb_A,rb_F,rb_H
  end
end

function online_phase(
  info::RBInfo,
  fe_sol::Vector{Snapshots},
  basis::RBSpace,
  op::Vector{<:RBVarOperator},
  tt::TimeTracker,
  args...)

  uh,μ = fe_sol
  rbopA,rbopF,rbopH = op
  rb_A,rb_F,rb_H = args

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      get_parameter(rbopA,μ[k],rb_A)
      get_parameter(rbopF,μ[k],rb_F)
      get_parameter(rbopH,μ[k],rb_H)
      lhs = assemble_rb_system(rbopA,rb_A,μ[k])
      rhs = assemble_rb_system(rbopF,rb_F,μ[k]),assemble_rb_system(rbopH,rb_H,μ[k])
      sys = poisson_rb_system(lhs,rhs)
      rb_sol = solve_rb_system(sys...)
    end
    uh_rb = reconstruct_fe_sol(basis,rb_sol)
    ErrorTracker(:u,uh,uh_rb,k)
  end

  ets = online_loop.(info.online_snaps)
  res = RBResults(:u,tt,ets)
  save(info,res)

  if info.postprocess
    postprocess(info,res)
  end
end
