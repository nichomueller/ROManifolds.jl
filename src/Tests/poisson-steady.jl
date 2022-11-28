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
  degree = 2

  ranges = Param.(fill([1.,10.],6))
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(mesh,ptype,root)
  model,dΩ,dΓn = model_info(bnd_info,degree,ptype)

  a,afe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,dΩ,dΓn)

  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamAffineFEOperator(lhs,rhs,PS,U,V)

  solver = LinearFESolver()
  uh,μ = get_fe_snapshots(solver,op,fepath,execute_fem,1)

  opA = ParamVarOperator(a,afe,PS,U,V,Nonaffine())
  opF = ParamVarOperator(f,ffe,PS,V,Nonaffine())
  opH = ParamVarOperator(h,hfe,PS,V,Nonaffine())

  ϵ = 1e-5
  use_energy_norm = false
  mdeim_nsnap = 20
  online_rhs = false
  get_offline_structures = false
  save_offline = true
  save_online = true
  postprocess = false
end
