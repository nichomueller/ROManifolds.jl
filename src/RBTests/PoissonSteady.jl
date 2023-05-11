root = pwd()

@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

function poisson_steady()
  run_fem = true

  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  test_path = "$root/tests/poisson/$mesh"
  mesh = "model.json"
  bnd_info = Dict("dirichlet" => ["sides","sides_c"],
                  "neumann" => ["circle","triangle","square"])
  order = 1

  ranges = fill([1.,20.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTrialFESpace(V,g)

  op = ParamAffineFEOperator(lhs,rhs,PS,U,V)

  solver = LinearFESolver()
  nsnap = 1
  uh,μ, = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamOperator(f,ffe,PS,V;id=:F)
  opH = NonaffineParamOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,test_path;ϵ=1e-4,nsnap=80,mdeim_snap=20,load_offline=false)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,online_structures = offline_phase(info,(uh,μ),(opA,opF,opH),measures,tt)
  online_phase(info,(uh,μ),rbspace,online_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::Vector{<:ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker)

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh,μ = fesol
  uh_offline = uh[1:info.nsnap]
  opA,opF,opH = op

  rbspace = assemble_rbspace(info,tt,uh_offline)

  rbopA = RBVariable(opA,rbspace,rbspace)
  rbopF = RBVariable(opF,rbspace)
  rbopH = RBVariable(opH,rbspace)

  Arb = RBAffineDecomposition(info,tt,rbopA,μ,meas,:dΩ)
  Frb = RBAffineDecomposition(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBAffineDecomposition(info,tt,rbopH,μ,meas,:dΓn)

  ad = (Arb,Frb,Hrb)
  ad_eval = eval_affine_decomposition(ad)
  online_structures = RBParamOnlineStructure(ad,ad_eval)

  rbspace,online_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::RBSpace,
  online_structures::Tuple,
  tt::TimeTracker)

  printstyled("Online phase, reduced basis method\n";color=:red)

  uh,μ = fesol

  function online_loop(k::Int)
    printstyled("-------------------------------------------------------------\n")
    printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
    tt.online_time += @elapsed begin
      lhs,rhs = steady_poisson_rb_system(online_structures,μ[k])
      rb_sol = solve_rb_system(lhs,rhs)
    end
    uhk = get_snap(uh[k])
    uhk_rb = reconstruct_fe_sol(rbspace,rb_sol)
    ErrorTracker(:u,uhk,uhk_rb)
  end

  ets = online_loop.(info.online_snaps)
  res = RBResults(:u,tt,ets)
  save(info,res)
  printstyled("Average online wall time: $(tt.online_time/length(ets_u)) s";
    color=:red)

  if info.postprocess
    postprocess(info,res)
  end
end

poisson_steady()
