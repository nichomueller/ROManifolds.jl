include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function navier_stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes"
  mesh = "flow_3cyl.json"
  bnd_info = Dict("dirichlet0" => ["noslip","cylinder"],
                  "dirichlet1" => ["nopenetration"],
                  "dirichlet2" => ["inlet","inlet_c"],
                  "neumann" => ["outlet"])
  dim = 3
  order = 2

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = [[1.,10.],[0.5,1.],[0.5,1.],[1.,2.]]
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,b,bfe,bTfe,c,cfe,d,dfe,f,ffe,h,hfe,g,res,jac,jac_t =
    navier_stokes_functions(ptype,measures)
  g0(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  g0(p::Param,t::Real) = x->g0(x,p,t)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{dim,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,
    dirichlet_tags=["dirichlet0","dirichlet1","dirichlet2"],
    dirichlet_masks=[(true,true,true),(false,false,true),(true,true,true)])
  U = ParamTransientTrialFESpace(V,[g0,g,g])
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)
  Y = ParamTransientMultiFieldFESpace([V,Q])
  X = ParamTransientMultiFieldFESpace([U,P])

  op = ParamTransientFEOperator(res,jac,jac_t,PS,X,Y)

  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = ThetaMethod(nls,dt,θ)
  nsnap = 100
  uh,ph,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,time_info,U,Q;id=:B)
  opBT = AffineParamOperator(b,bTfe,PS,time_info,P,V;id=:BT)
  opC = NonlinearParamOperator(c,cfe,PS,time_info,U,V;id=:C)
  opD = NonlinearParamOperator(d,dfe,PS,time_info,U,V;id=:D)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

  for fun_mdeim = (false)
    for st_mdeim = (true)
      for tol = (1e-1,1e-2,1e-3,1e-4)

        global info = RBInfoUnsteady(ptype,mesh,root;ϵ=tol,nsnap=80,online_snaps=95:100,
          mdeim_snap=15,load_offline=false,postprocess=true,
          fun_mdeim=fun_mdeim,st_mdeim=st_mdeim)
        tt = TimeTracker(OfflineTime(0.,0.),0.)

        printstyled("Offline phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:blue)

        μ_offline = μ[1:info.nsnap]
        uh_offline = uh[1:info.nsnap]
        ph_offline = ph[1:info.nsnap]
        uhθ_offline = compute_in_timesθ(uh_offline,θ)
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
