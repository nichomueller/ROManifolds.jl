include("tests.jl")

function configure()
  I=true
  S=true
  M=false
  ptype = ProblemType{I,S,M}()
  execute_fem = true

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  degree = 2
  sampling = UniformSampling()
  ranges = Param.(fill([1.,10.],6))

  fepath,model,dΩ,dΓn,PS =
    configure(ptype,degree,ranges;root,mesh,bnd_info,sampling)

  afe,bfe,cfe,dfe,ffe,hfe,aμ,bμ,cμ,dμ,fμ,hμ,gμ,res,jac =
    navier_stokes_functions(ptype,dΩ,dΓn,PS)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,degree-1;space=:P)

  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,gμ)
  Q = MyTests(model,reffe2;conformity=:L2)
  P = MyTrials(Q)

  X = ParamMultiFieldFESpace([U,P])
  Y = ParamMultiFieldFESpace([V,Q])
  op = ParamFEOperator(res,jac,jac_t,PS,X,Y)
  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = FESolver(nls)
  uh,μ = get_fe_snapshots(solver,op,fepath,execute_fem,1)

  opA = ParamVarOperator(aμ,afe,U,V,Nonaffine())
  opB = ParamVarOperator(bμ,bfe,U,Q,Affine())
  opC = ParamVarOperator(cμ,cfe,U,V,Nonlinear())
  opD = ParamVarOperator(dμ,dfe,U,V,Nonlinear())
  opF = ParamVarOperator(fμ,ffe,V,Nonaffine())
  opH = ParamVarOperator(hμ,hfe,V,Nonaffine())

  Problem(ptype,μ,uh,[opA,opB,opC,opD,opM,opF,opH])
end

const feproblem = configure()
