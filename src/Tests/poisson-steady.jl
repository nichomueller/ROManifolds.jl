include("tests.jl")

function configure()
  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)
  execute_fem = true

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1

  ranges = Param.(fill([1.,10.],6))
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  model = model_info(bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamAffineFEOperator(lhs,rhs,PS,U,V)

  solver = LinearFESolver()
  uh,ph,μ = get_fe_snapshots(solver,op,fepath,execute_fem,1)

  opA = ParamVarOperator(a,afe,PS,U,V,Nonaffine())
  opF = ParamVarOperator(f,ffe,PS,V,Nonaffine())
  opH = ParamVarOperator(h,hfe,PS,V,Nonaffine())

  rbinfo = RBInfo(ptype,mesh,root;ϵ=1e-5,mdeim_nsnap=20)

  offline_phase()
  online_phase()
end

function offline_phase()
  if load_offline
    get_rb(RBInfo,RBVars)
    operators = load_offline(RBInfo,RBVars)
    if !all(isempty.(operators))
      assemble_offline_structures(RBInfo,RBVars,operators)
    end
  else
    println("Building reduced basis via POD")
    assemble_rb.(uh,ph)
    operators = set_operators(RBInfo)
    assemble_offline_structures(RBInfo,RBVars,operators)
  end
end

function online_phase()

end
