include("../../../src/Utils/Utils.jl")
include("../../../src/FEM/FEM.jl")
include("../../../src/FEM/NewFESpaces.jl")
include("../../../src/FEM/NewTypes.jl")
include("../../../src/FEM/NewFEOperators.jl")
include("../../../src/FEM/ParamOperatorInterfaces.jl")
include("../../../src/FEM/NewFESolvers.jl")

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
P = ParamSpace(ranges,UniformSampling())

a(μ::Param,x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
a(μ::Param) = x->a(μ,x)
b(μ::Param,x) = 1.
b(μ::Param) = x->b(μ,x)
f(μ::Param,x) = 1. + Point(μ[4:6]) .* x
f(μ::Param) = x->f(μ,x)
h(μ::Param,x) = 1. + Point(μ[4:6]) .* x
h(μ::Param) = x->h(μ,x)
g(μ::Param,x) = 1. + Point(μ[4:6]) .* x
g(μ::Param) = x->g(μ,x)

afe(μ,u,v) = ∫(a(μ) * ∇(v) ⊙ ∇(u))dΩ
bfe(μ,u,q) = ∫(b(μ) * q * (∇⋅(u)))dΩ
ffe(μ,v) = ∫(f(μ) ⋅ v)dΩ
hfe(μ,v) = ∫(h(μ) ⋅ v)dΓn

#lhs(μ,(u,p),(v,q)) = afe(μ,u,v) - bfe(μ,p,v) - bfe(μ,u,q)
rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
lhs(μ,(u,p),(v,q)) = ∫(a(μ)*∇(v)⊙∇(u) - b(μ)*((∇⋅v)*p + q*(∇⋅u)))dΩ

reffe1 = Gridap.ReferenceFE(lagrangian, VectorValue{3,Float}, 2)
reffe2 = Gridap.ReferenceFE(lagrangian, Float, 1; space=:P)

I=true
S=true

Gμ = ParamFunction(P,g;S)
myV = MyTests(model, reffe1; conformity=:H1, dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)
myQ = MyTests(model, reffe2; conformity=:L2)
myP = MyTrials(myQ)

X = ParamMultiFieldTrialFESpace([myU.trial,myP.trial])
Y = MultiFieldFESpace([myV.test,myQ.test])
op = ParamAffineFEOperator(lhs,rhs,P,X,Y)
ye = solve(LinearFESolver(),op,1)

opA = ParamVarOperator(a,afe,P,myU,myV)
opB = ParamVarOperator(b,bfe,P,myU,myQ)
opF = ParamVarOperator(f,ffe,P,myV)
opH = ParamVarOperator(h,hfe,P,myV)
stokes_problem = Problem(μ,[uh,ph],[opA,opB,opF,opH])
