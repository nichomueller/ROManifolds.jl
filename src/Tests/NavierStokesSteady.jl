include("tests.jl")

function configure()
  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)
  run_fem = true

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  degree = 2

  ranges = Param.(fill([1.,10.],6))
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(mesh,ptype,root)
  model,dΩ,dΓn = model_info(bnd_info,degree,ptype)

  a,b,f,h,g,afe,bfe,cfe,dfe,ffe,hfe,res,jac =
    navier_stokes_functions(dΩ,dΓn,ptype)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,degree-1;space=:P)

  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  Q = MyTests(model,reffe2;conformity=:L2)
  P = MyTrials(Q)

  X = ParamMultiFieldFESpace([U,P])
  Y = ParamMultiFieldFESpace([V,Q])
  op = ParamFEOperator(res,jac,PS,X,Y)
  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = FESolver(nls)
  uh,μ = get_fe_snapshots(solver,op,fepath,run_fem,1)

  opA = ParamVarOperator(a,afe,U,V,Nonaffine())
  opB = ParamVarOperator(b,bfe,U,Q,Affine())
  opC = ParamVarOperator(c,cfe,U,V,Nonlinear())
  opD = ParamVarOperator(d,dfe,U,V,Nonlinear())
  opF = ParamVarOperator(f,ffe,V,Nonaffine())
  opH = ParamVarOperator(h,hfe,V,Nonaffine())

  Problem(ptype,μ,uh,[opA,opB,opC,opD,opM,opF,opH])
end

const feproblem = configure()
