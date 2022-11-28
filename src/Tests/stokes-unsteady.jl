include("../../../src/FEM/FEM.jl")
include("../../../src/FEM/NewFESpaces.jl")
include("../../../src/FEM/NewTypes.jl")
include("../../../src/FEM/NewFEOperators.jl")
include("../../../src/FEM/ParamOperatorInterfaces.jl")
include("../../../src/FEM/NewFESolvers.jl")
include("../../../src/FEM/TransientFESolutions.jl")
include("../../../src/FEM/AffineThetaMethod.jl")
include("../../../src/FEM/ThetaMethod.jl")
include("../../../src/FEM/DiffOperators.jl")

root = "/home/nicholasmueller/git_repos/Mabla.jl"
mesh_name = "cube5x5x5.json"
model = DiscreteModelFromFile(joinpath(root, "tests/meshes/$mesh_name"))

function set_labels!(bnd_info::Dict)
  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)
  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels, tags[i], bnds[i])
    end
  end
end
bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])
set_labels!(bnd_info)

degree=2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γn = BoundaryTriangulation(model, tags=["neumann"])
dΓn = Measure(Γn, degree)

ranges = [[1., 10.], [1., 10.], [1., 10.],
          [1., 10.], [1., 10.], [1., 10.]]
pspace = ParamSpace(ranges,UniformSampling())

a(x,t::Real,μ::Param) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
a(t::Real,μ::Param) = x->a(x,t,μ)
a(μ::Param) = t->a(t,μ)
b(x,t::Real,μ::Param) = 1.
b(t::Real,μ::Param) = x->b(x,t,μ)
b(μ::Param) = t->b(t,μ)
f(x,t::Real,μ::Param) = 1. + sin(t)*Point(μ[4:6]).*x
f(t::Real,μ::Param) = x->f(x,t,μ)
f(μ::Param) = t->f(t,μ)
h(x,t::Real,μ::Param) = 1. + sin(t)*Point(μ[4:6]).*x
h(t::Real,μ::Param) = x->h(x,t,μ)
h(μ::Param) = t->h(t,μ)
g(x,t::Real,μ::Param) = 1. + sin(t)*Point(μ[4:6]).*x
g(t::Real,μ::Param) = x->g(x,t,μ)
g(μ::Param) = t->g(t,μ)

mfe(μ,t,(u,p),(v,q)) = ∫(v ⋅ u)dΩ
afe(μ,t,u,v) = ∫(a(t,μ) * ∇(v) ⊙ ∇(u))dΩ
bfe(μ,t,u,q) = ∫(b(t,μ) * q * (∇⋅(u)))dΩ
ffe(μ,t,v) = ∫(f(t,μ) ⋅ v)dΩ
hfe(μ,t,v) = ∫(h(t,μ) ⋅ v)dΓn

rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
lhs(μ,t,(u,p),(v,q)) = ∫(a(t,μ)*∇(v)⊙∇(u) - b(t,μ)*((∇⋅v)*p + q*(∇⋅u)))dΩ

reffe1 = Gridap.ReferenceFE(lagrangian, VectorValue{3,Float}, 2)
reffe2 = Gridap.ReferenceFE(lagrangian, Float, 1; space=:P)

I=true
S=false

Gμ = ParamFunction(pspace,g;S)
myV = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)
myQ = MyTests(model,reffe2; conformity=:L2)
myP = MyTrials(myQ)

X = ParamTransientMultiFieldFESpace([myU.trial,myP.trial])
Y = ParamTransientMultiFieldFESpace([myV.test,myQ.test])
op = ParamTransientAffineFEOperator(mfe,lhs,rhs,pspace,X,Y)
ode_solver = ThetaMethod(LUSolver(),0.025,0.5)
ye = solve(ode_solver,op,0.,0.5)
count = 1
for (xₕ, _) in ye[1]
  println("Time step: $count")
  get_free_dof_values(xₕ)
  count += 1
end

opA = ParamVarOperator(a,afe,pspace,myU,myV)
opB = ParamVarOperator(b,bfe,pspace,myU,myQ)
opF = ParamVarOperator(f,ffe,pspace,myV)
opH = ParamVarOperator(h,hfe,pspace,myV)

stokes_problem = Problem(μ,[uh,ph],[opA,opB,opF,opH])
