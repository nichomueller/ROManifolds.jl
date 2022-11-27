include("../../../FEM/FEM.jl")
include("../../tests.jl")

function configure()
  I=true
  S=false

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  sampling = UniformSampling()
  degree = 2
  ranges = fill([1.,10.],6)

  fepath,model,dΩ,dΓn,PS =
    configure(degree,ranges;root,mesh,bnd_info,sampling,S)

  a(x,μ::Vector{Float},t::Real) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
  a(μ::Vector{Float},t::Real) = x->a(x,μ,t)
  a(μ::Vector{Float}) = t->a(μ,t)
  b(x,μ::Vector{Float},t::Real) = 1.
  b(μ::Vector{Float},t::Real) = x->b(x,μ,t)
  b(μ::Vector{Float}) = t->b(μ,t)
  f(x,μ::Vector{Float},t::Real) = 1. + sin(t)*Point(μ[4:6]).*x
  f(μ::Vector{Float},t::Real) = x->f(x,μ,t)
  f(μ::Vector{Float}) = t->f(μ,t)
  h(x,μ::Vector{Float},t::Real) = 1. + sin(t)*Point(μ[4:6]).*x
  h(μ::Vector{Float},t::Real) = x->h(x,μ,t)
  h(μ::Vector{Float}) = t->h(μ,t)
  g(x,μ::Vector{Float},t::Real) = 1. + sin(t)*Point(μ[4:6]).*x
  g(μ::Vector{Float},t::Real) = x->g(x,μ,t)
  g(μ::Vector{Float}) = t->g(μ,t)

  mfe(μ,t,u,v) = ∫(v⋅u)dΩ
  afe(μ,t,u,v) = ∫(a(μ,t)*∇(v) ⊙ ∇(u))dΩ
  bfe(μ,t,u,q) = ∫(b(μ,t)*q*(∇⋅(u)))dΩ
  cfe(μ,u,v) = ∫(v⊙(∇(u)'⋅μ))dΩ
  cfe(μ) = (u,v) -> cfe(μ,u,v)
  dfe(μ,u,v) = ∫(v⊙(∇(μ)'⋅u))dΩ
  dfe(μ) = (u,v) -> dfe(μ,u,v)
  ffe(μ,t,v) = ∫(f(μ,t)⋅v)dΩ
  hfe(μ,t,v) = ∫(h(μ,t)⋅v)dΓn
  #FOR SOME REASON, BETTER CONVERGENCE WITH THIS DEF OF NONLINEAR TERM
  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  res(μ,t,(u,p),(v,q)) = mfe(μ,t,(∂t(u),∂t(p)),(v,q)) + lhs(μ,t,(u,p),(v,q)) + c(u,v) - rhs(μ,t,(v,q))
  jac(μ,t,(u,p),(du,dp),(v,q)) = lhs(μ,t,(du,dp),(v,q)) + dc(u,du,v)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = mfe(μ,t,(dut,dpt),(v,q))

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,degree-1;space=:P)

  Gμ = ParamFunctional(PS,g;S)
  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,Gμ)
  Q = MyTests(model,reffe2;conformity=:L2)
  P = MyTrials(Q)

  dt,t0,tF,θ = 0.025,0.,0.05,0.5

  X = ParamTransientMultiFieldFESpace([U,P])
  Y = ParamTransientMultiFieldFESpace([V,Q])
  op = ParamTransientFEOperator(res,jac,jac_t,PS,X,Y)
  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = ThetaMethod(nls,dt,θ)
  uh,μ = run(solver,op,t0,tF,1)
  save(uh,fepath),save(μ,fepath)

  opA = ParamVarOperator(a,afe,PS,U,V,Nonaffine())
  opB = ParamVarOperator(b,bfe,PS,U,Q,Affine())
  opC = ParamVarOperator(cfe,cfe,PS,U,V,Nonlinear())
  opD = ParamVarOperator(dfe,dfe,PS,U,V,Nonlinear())
  opF = ParamVarOperator(f,ffe,PS,V,Nonaffine())
  opH = ParamVarOperator(h,hfe,PS,V,Nonaffine())
  opM = ParamVarOperator(mfe,mfe,PS,U,V,Affine())

  Problem(μ,uh,[opA,opB,opC,opD,opM,opF,opH],t0,tF,dt,θ,I)
end

feproblem = configure()
