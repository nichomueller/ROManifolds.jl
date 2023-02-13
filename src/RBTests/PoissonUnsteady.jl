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
  mesh = "model.json"
  bnd_info = Dict("dirichlet" => ["sides","sides_c"],
                  "neumann" => ["circle","triangle","square"])
  order = 1

  t0,tF,dt,θ = 0.,0.3,0.005,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,10.],4)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTransientTrialFESpace(V,g)
  op = ParamTransientAffineFEOperator(mfe,lhs,rhs,PS,U,V)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

  for fun_mdeim = (true)#(false,true)
    for st_mdeim = (false,true)
      for tol = (1e-2,1e-3,1e-4,1e-5)

        info = RBInfoUnsteady(ptype,mesh,root;ϵ=tol,nsnap=80,online_snaps=95:100,
          mdeim_snap=20,load_offline=false,postprocess=true,
          fun_mdeim=fun_mdeim,st_mdeim=st_mdeim)
        tt = TimeTracker(OfflineTime(0.,0.),0.)

        printstyled("Offline phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:blue)

        uh_offline = uh[1:info.nsnap]
        rbspace = rb(info,tt,uh_offline)

        rbopA = RBVariable(opA,rbspace,rbspace)
        rbopM = RBVariable(opM,rbspace,rbspace)
        rbopF = RBVariable(opF,rbspace)
        rbopH = RBVariable(opH,rbspace)

        Arb = RBAffineDecomposition(info,tt,rbopA,μ,measures,:dΩ)
        Mrb = RBAffineDecomposition(info,tt,rbopM,μ,measures,:dΩ)
        Frb = RBAffineDecomposition(info,tt,rbopF,μ,measures,:dΩ)
        Hrb = RBAffineDecomposition(info,tt,rbopH,μ,measures,:dΓn)

        ad = (Arb,Mrb,Frb,Hrb)
        ad_eval = eval_affine_decomposition(ad)
        param_on_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

        printstyled("Online phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:red)

        function online_loop(k::Int)
          printstyled("-------------------------------------------------------------\n")
          printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
          tt.online_time += @elapsed begin
            printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
            lhs,rhs = unsteady_poisson_rb_system(param_on_structures,μ[k])
            rb_sol = solve_rb_system(lhs,rhs)
          end
          uhk = get_snap(uh[k])
          uhk_rb = reconstruct_fe_sol(rbspace,rb_sol)
          ErrorTracker(:u,uhk,uhk_rb)
        end

        ets = online_loop.(info.online_snaps)
        res = RBResults(:u,tt,ets)
        save(info,res)
        printstyled("Average online wall time: $(tt.online_time/length(ets)) s";
          color=:red)

        if info.postprocess
          trian = get_triangulation(model)
          k = rand(info.online_snaps)
          writevtk(info,time_info,uh[k],t->U(μ[k],t),trian)
          writevtk(info,time_info,res,V,trian)
        end
      end
    end
  end

  if info.postprocess
    postprocess(info)
  end
end

poisson_unsteady()
