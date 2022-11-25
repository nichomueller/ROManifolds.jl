include("../../../src/Utils/Utils.jl")
include("../../../src/FEM/FEM.jl")
include("../../../src/FEM/NewTypes1.jl")
include("../../../src/FEM/NewFESpaces.jl")
include("../../../src/FEM/NewFEOperators.jl")
include("../../../src/FEM/ParamOperatorInterfaces.jl")
include("../../../src/FEM/NewFESolvers.jl")

root = "/home/nicholasmueller/git_repos/Mabla.jl"
mesh_name = "cube5x5x5.json"
model = DiscreteModelFromFile(joinpath(root,"tests/meshes/$mesh_name"))

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

degree=1
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γn = BoundaryTriangulation(model, tags=["neumann"])
dΓn = Measure(Γn, degree)

ranges = [[1., 10.], [1., 10.], [1., 10.],
          [1., 10.], [1., 10.], [1., 10.]]
P = ParamSpace(ranges,UniformSampling())

a(μ::Vector{Float},x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
a(μ::Vector{Float}) = x->a(μ,x)
f(μ::Vector{Float},x) = sum(Point(μ[4:6]) .* x)
f(μ::Vector{Float}) = x->f(μ,x)
h(μ::Vector{Float},x) = sum(Point(μ[4:6]) .* x)
h(μ::Vector{Float}) = x->h(μ,x)
g(μ::Vector{Float},x) = sum(Point(μ[4:6]) .* x)
g(μ::Vector{Float}) = x->g(μ,x)

afe(μ,u,v) = ∫(a(μ) * ∇(v) ⋅ ∇(u))dΩ
ffe(μ,v) = ∫(f(μ) * v)dΩ
hfe(μ,v) = ∫(h(μ) * v)dΓn

lhs(μ,u,v) = afe(μ,u,v)
rhs(μ,v) = ffe(μ,v) + hfe(μ,v)

reffe = Gridap.ReferenceFE(lagrangian, Float, 1)

I=false
S=true

Gμ = ParamFunctional(P,g;S)
myV = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)

op = ParamAffineFEOperator(lhs,rhs,P,myU.trial,myV.test)
ye = solve(LinearFESolver(),op,1)

opA = ParamVarOperator(a,afe,P,myU,myV)
opF = ParamVarOperator(f,ffe,P,myV)
opH = ParamVarOperator(h,hfe,P,myV)
poisson_problem = Problem(μ,[uh],[opA,opF,opH])
