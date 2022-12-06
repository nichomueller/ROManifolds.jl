include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("tests.jl")

function poisson_steady()
  steady = false
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)
  run_fem = true

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1
  degree = get_degree(order)

  ranges = fill([1.,10.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  model = model_info(bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,f,ffe,h,hfe,g,mfe,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamTransientAffineFEOperator(mfe,lhs,rhs,PS,U.trial,V.test)

  solver = ThetaMethod(LUSolver(),0.025,0.5)
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,0.,0.5,100)

  opA = ParamVarOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = ParamVarOperator(h,hfe,PS,V;id=:H)

  rbinfo = RBInfo(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_nsnap=20)

  tt,basis,rb_A,rb_F,rb_H = offline_phase(rbinfo,[uh,μ],[opA,opF,opH],measures)
  online_phase(info,[uh,μ],basis,tt,rb_A,rb_F,rb_H)
end
