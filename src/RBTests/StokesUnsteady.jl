root = pwd()

using Distributed
@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

function stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  mesh = "flow_3cyl2D.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  order = 2

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)
  D = get_dimension(model)

  ranges = fill([1.,10.],4)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = TimeInfo(t0,tF,dt,θ)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1/sum(μ)
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  b(x,p::Param,t::Real) = 1.
  b(p::Param,t::Real) = x->b(x,p,t)
  m(x,p::Param,t::Real) = 1.
  m(p::Param,t::Real) = x->m(x,p,t)
  f(x,p::Param,t::Real) = VectorValue(0.,0.)
  f(p::Param,t::Real) = x->f(x,p,t)
  h(x,p::Param,t::Real) = VectorValue(0.,0.)
  h(p::Param,t::Real) = x->h(x,p,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    W = 1.5
    T = 0.16
    flow_rate = μ[4]*abs(1-cos(pi*t/T)+μ[2]*sin(μ[3]*pi*t/T))
    parab_prof = VectorValue(abs.(x[2]*(x[2]-W))/(W/2)^2,0.)
    parab_prof*flow_rate
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g0(x,p::Param,t::Real) = VectorValue(0.,0.)
  g0(p::Param,t::Real) = x->g0(x,p,t)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{D,Float},order)
  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{D,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  U = ParamTransientTrialFESpace(V,[g0,g])
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)

  feop,opA,opB,opBT,opM,opF,opH = stokes_operators(measures,PS,time_info,V,U,Q,P;a,b,m,f,h)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,ph,μ = fe_snapshots(ptype,solver,feop,fepath,run_fem,nsnap,t0,tF)

  info = RBInfoUnsteady(ptype,test_path;ϵ=1e-3,nsnap=80,mdeim_snap=10)
  tt = TimeTracker(OfflineTime(0.,0.),0.)

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]

  tt.offline_time.basis_time += @elapsed begin
    rbspace_u,rbspace_p = rb(info,(uh_offline,ph_offline),opB,ph,μ)
  end
  rbspace = rbspace_u,rbspace_p

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
  rbopM = RBVariable(opM,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  tt.offline_time.assembly_time = @elapsed begin
    Arb = RBAffineDecomposition(info,rbopA,μ,measures,:dΩ)
    Brb = RBAffineDecomposition(info,rbopB,μ,measures,:dΩ)
    BTrb = RBAffineDecomposition(info,rbopBT,μ,measures,:dΩ)
    Mrb = RBAffineDecomposition(info,rbopM,μ,measures,:dΩ)
    Frb = RBAffineDecomposition(info,rbopF,μ,measures,:dΩ)
    Hrb = RBAffineDecomposition(info,rbopH,μ,measures,:dΓn)
  end

  ad = (Arb,Brb,BTrb,Mrb,Frb,Hrb)
  ad_eval = eval_affine_decomposition(ad)
  param_on_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

  save(info,(rbspace,ad))

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
  printstyled("Average online wall time: $(tt.online_time/length(ets_u)) s\n";
    color=:red)

  if info.postprocess
    postprocess(info)
    trian = get_triangulation(model)
    k = first(info.online_snaps)
    writevtk(info,time_info,uh[k],t->U(μ[k],t),trian)
    writevtk(info,time_info,ph[k],t->P,trian)
    writevtk(info,time_info,res_u,V,trian)
    writevtk(info,time_info,res_p,Q,trian)
  end
end

stokes_unsteady()
