include("tests.jl")

function configure()
  I=true
  S=false
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

  afe,bfe,cfe,dfe,ffe,hfe,mfe,aμ,bμ,cμ,dμ,fμ,hμ,gμ,mμ,res,jac,jac_t =
    navier_stokes_functions(ptype,dΩ,dΓn,PS)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,degree-1;space=:P)

  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,gμ)
  Q = MyTests(model,reffe2;conformity=:L2)
  P = MyTrials(Q)

  dt,t0,tF,θ = 0.025,0.,0.05,0.5

  X = ParamTransientMultiFieldFESpace([U,P])
  Y = ParamTransientMultiFieldFESpace([V,Q])
  op = ParamTransientFEOperator(res,jac,jac_t,PS,X,Y)
  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = ThetaMethod(nls,dt,θ)
  uh,μ = get_fe_snapshots(solver,op,fepath,execute_fem,t0,tF,1)

  opA = ParamVarOperator(aμ,afe,U,V,Nonaffine())
  opB = ParamVarOperator(bμ,bfe,U,Q,Affine())
  opC = ParamVarOperator(cμ,cfe,U,V,Nonlinear())
  opD = ParamVarOperator(dμ,dfe,U,V,Nonlinear())
  opF = ParamVarOperator(fμ,ffe,V,Nonaffine())
  opH = ParamVarOperator(hμ,hfe,V,Nonaffine())
  opM = ParamVarOperator(mμ,mfe,U,V,Affine())

  Problem(ptype,μ,uh,[opA,opB,opC,opD,opM,opF,opH],t0,tF,dt,θ)
end

const feproblem = configure()
