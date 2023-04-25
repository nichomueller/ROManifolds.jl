root = pwd()

@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

function navier_stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  test_path = "$root/tests/navier-stokes/$mesh"
  mesh = "flow_3cyl2D.json"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  dim = 2
  order = 2

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = [[1.,10.],[0.5,1.],[0.5,1.],[1.,2.]]
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1e-3*μ[1]
  end
  a(μ::Param,t::Real) = x->a(x,μ,t)
  b(x,μ::Param,t::Real) = 1.
  b(μ::Param,t::Real) = x->b(x,μ,t)
  c(x,μ::Param,t::Real) = 1.
  c(μ::Param,t::Real) = x->c(x,μ,t)
  d(x,μ::Param,t::Real) = 1.
  d(μ::Param,t::Real) = x->d(x,μ,t)
  m(x,μ::Param,t::Real) = 1.
  m(μ::Param,t::Real) = x->m(x,μ,t)
  f(x,p::Param,t::Real) = VectorValue(0.,0.)
  f(μ::Param,t::Real) = x->f(x,μ,t)
  h(x,p::Param,t::Real) = VectorValue(0.,0.)
  h(μ::Param,t::Real) = x->h(x,μ,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    W = 1.5
    T = 0.16
    flow_rate = μ[4]*abs(1-cos(pi*t/T)+μ[2]*sin(μ[3]*pi*t/T))
    parab_prof = VectorValue(abs.(x[2]*(x[2]-W))/(W/2)^2,0.)
    parab_prof*flow_rate
  end
  g(μ::Param,t::Real) = x->g(x,μ,t)
  g0(x,p::Param,t::Real) = VectorValue(0.,0.)
  g0(p::Param,t::Real) = x->g0(x,p,t)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{dim,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  U = ParamTransientTrialFESpace(V,[g0,g,g])
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)

  feop,opA,opB,opBT,opC,opD,opM,opF,opH =
    navier_stokes_operators(measures,PS,time_info,V,U,Q,P;a,b,c,d,m,f,h)

  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = ThetaMethod(nls,dt,θ)
  nsnap = 100
  uh,ph,μ = fe_snapshots(ptype,solver,feop,fepath,run_fem,nsnap,t0,tF)

  for fun_mdeim = (false)
    for st_mdeim = (true)
      for tol = (1e-1,1e-2,1e-3,1e-4)

        global info = RBInfoUnsteady(ptype,test_path;ϵ=tol,nsnap=80,online_snaps=95:100,
          mdeim_snap=15,load_offline=false,postprocess=true,
          fun_mdeim=fun_mdeim,st_mdeim=st_mdeim)
        tt = TimeTracker(OfflineTime(0.,0.),0.)

        printstyled("Offline phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:blue)

        μ_offline = μ[1:info.nsnap]
        uh_offline = uh[1:info.nsnap]
        ph_offline = ph[1:info.nsnap]
        uhθ_offline = compute_in_times(uh_offline,θ)
        ghθ_offline = get_dirichlet_values(U,μ_offline,time_info)
        ughθ_offline = vcat(uhθ_offline,ghθ_offline)

        rbspace_u,rbspace_p = rb(info,(uh_offline,ph_offline),opB,ph,μ;tt)
        rbspace = rbspace_u,rbspace_p

        rbopA = RBVariable(opA,rbspace_u,rbspace_u)
        rbopB = RBVariable(opB,rbspace_p,rbspace_u)
        rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
        rbopC = RBVariable(opC,rbspace_u,rbspace_u)
        rbopD = RBVariable(opD,rbspace_u,rbspace_u)
        rbopM = RBVariable(opM,rbspace_u,rbspace_u)
        rbopF = RBVariable(opF,rbspace_u)
        rbopH = RBVariable(opH,rbspace_u)

        Arb = RBAffineDecomposition(info,tt,rbopA,μ,measures,:dΩ)
        Brb = RBAffineDecomposition(info,tt,rbopB,μ,measures,:dΩ)
        BTrb = RBAffineDecomposition(info,tt,rbopBT,μ,measures,:dΩ)
        Crb = RBAffineDecomposition(info,tt,rbopC,μ,measures,:dΩ,ughθ_offline)
        Drb = RBAffineDecomposition(info,tt,rbopD,μ,measures,:dΩ,ughθ_offline)
        Mrb = RBAffineDecomposition(info,tt,rbopM,μ,measures,:dΩ)
        Frb = RBAffineDecomposition(info,tt,rbopF,μ,measures,:dΩ)
        Hrb = RBAffineDecomposition(info,tt,rbopH,μ,measures,:dΓn)

        ad = (Arb,Brb,BTrb,Crb,Drb,Mrb,Frb,Hrb)
        ad_eval = eval_affine_decomposition(ad)
        param_on_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

        save(info,(rbspace,ad))

        printstyled("Online phase; tol=$tol, st_mdeim=$st_mdeim\n";color=:red)

        rb_solver(res,jac,x0,Uk) = solve_rb_system(res,jac,x0,Uk,rbspace,time_info;tol=info.ϵ)

        function online_loop(k::Int)
          printstyled("-------------------------------------------------------------\n")
          printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
          tt.online_time += @elapsed begin
            res,jac = unsteady_navier_stokes_rb_system(param_on_structures,μ[k])
            Uk = U(μ[k])
            x0 = initial_guess(uh,ph,μ_offline,μ[k])
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
        printstyled("Average online wall time: $(tt.online_time/length(ets_u)) s\n";
          color=:red)

        if info.postprocess
          trian = get_triangulation(model)
          k = first(info.online_snaps)
          writevtk(info,time_info,uh[k],t->U(μ[k],t),trian)
          writevtk(info,time_info,ph[k],t->P,trian)
          writevtk(info,time_info,res_u,V,trian)
          writevtk(info,time_info,res_p,Q,trian)
        end
      end
    end
  end

  if info.postprocess
    postprocess(info)
  end

end

navier_stokes_unsteady()
