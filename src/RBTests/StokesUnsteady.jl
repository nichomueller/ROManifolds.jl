include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function stokes_unsteady()
  run_fem = true

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/stokes"
  mesh = "bifurcation_coarse.json"
  bnd_info = Dict("dirichlet0" => [4],"dirichlet_in1" => [2],"dirichlet_in2" => [3],"neumann" => [1])
  order = 2

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,3.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,jac_t,b,bfe,bTfe,f,ffe,h,hfe,g,lhs,rhs = stokes_functions(ptype,measures)
  g0(x,p::Param,t::Real) = VectorValue(0,0,0)
  g0(p::Param,t::Real) = x->g0(x,p,t)
  gout(x,p::Param,t::Real) = 0.3*g(x,p,t)
  gout(p::Param,t::Real) = x->gout(x,p,t)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet_in1","dirichlet_in2"])
  U = ParamTransientTrialFESpace(V,[g0,g,gout])
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)
  Y = ParamTransientMultiFieldFESpace([V,Q])
  X = ParamTransientMultiFieldFESpace([U,P])

  op = ParamTransientAffineFEOperator(jac_t,lhs,rhs,PS,X,Y)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 1
  uh,ph,μ, = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)
#=
  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,time_info,U,Q;id=:B)
  opBT = AffineParamOperator(b,bTfe,PS,time_info,P,V;id=:BT)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-3,nsnap=80,mdeim_snap=5,load_offline=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh,ph,μ = fesol
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  opA,opB,opBT,opM,opF,opH = op

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)
  rbspace = rbspace_u,rbspace_p

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
  rbopM = RBVariable(opM,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  Arb = RBAffineDecomposition(info,tt,rbopA,μ,measures,:dΩ)
  Brb = RBAffineDecomposition(info,tt,rbopB,μ,measures,:dΩ)
  BTrb = RBAffineDecomposition(info,tt,rbopBT,μ,measures,:dΩ)
  Mrb = RBAffineDecomposition(info,tt,rbopM,μ,measures,:dΩ)
  Frb = RBAffineDecomposition(info,tt,rbopF,μ,measures,:dΩ)
  Hrb = RBAffineDecomposition(info,tt,rbopH,μ,measures,:dΓn)

  ad = (Arb,Brb,BTrb,Mrb,Frb,Hrb)
  ad_eval = eval_affine_decomposition(ad)
  param_on_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

  printstyled("Online phase, reduced basis method\n";color=:red)

  function online_loop(k::Int)
    printstyled("-------------------------------------------------------------\n")
    printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
    tt.online_time += @elapsed begin
      lhs,rhs = unsteady_stokes_rb_system(param_on_structures,μ[k])
      rb_sol = solve_rb_system(lhs,rhs)
    end
    uhk = get_snap(uh[k])
    phk = get_snap(ph[k])
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
    postprocess(info)
  end =#
end

stokes_unsteady()
