using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.FESpaces
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB
using LinearAlgebra
using SparseArrays

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

for n in (5,10,20)
  domain = (0,1,0,1)
  partition = (n,n)
  model = CartesianDiscreteModel(domain, partition)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
  add_tag_from_tags!(labels,"neumann",[7])

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  r = realization(pspace)
  μ = r.params[1]
  t = rand(1)[1]

  a(x) = 1+exp(-sin(t)^2*x[1]/sum(μ))
  f(x) = 1.
  h(x) = abs(cos(t/μ[3]))
  g(x) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))

  b(v) = ∫(f*v)dΩ + ∫(h*v)dΓn
  a(du,v) = ∫(du*v)dΩ + ∫(a*∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = TrialFESpace(test,g)
  feop = AffineFEOperator(a,b,trial,test)
  fesolver = LUSolver()
  uh = solve(fesolver,feop)

  plot_dir = datadir(joinpath("heateq","plot_$(n)"))
  writevtk(Ω,plot_dir,cellfields=["uh"=>uh])

  println(t)
  println(μ)
end
