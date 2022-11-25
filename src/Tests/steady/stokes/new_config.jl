include("../../../src/Utils/Utils.jl")
include("../../../src/FEM/FEM.jl")
include("../../../src/FEM/NewFESpaces.jl")
include("../../../src/FEM/NewTypes.jl")
include("../../../src/FEM/NewFEOperators.jl")
include("../../../src/FEM/ParamOperatorInterfaces.jl")
include("../../../src/FEM/OperatorsFromProblem.jl")

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

ranges = [[1., 10.], [1., 10.], [1., 10.],
          [1., 10.], [1., 10.], [1., 10.]]
P = ParamSpace(ranges,UniformSampling())

a(μ::Vector{Float},x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
a(μ::Vector{Float}) = x->a(μ,x)
b(μ::Vector{Float},x) = 1.
b(μ::Vector{Float}) = x->b(μ,x)
f(μ::Vector{Float},x) = 1. + Point(μ[4:6]) .* x
f(μ::Vector{Float}) = x->f(μ,x)
h(μ::Vector{Float},x) = 1. + Point(μ[4:6]) .* x
h(μ::Vector{Float}) = x->h(μ,x)
g(μ::Vector{Float},x) = 1. + Point(μ[4:6]) .* x
g(μ::Vector{Float}) = x->g(μ,x)

afe(a::Function,u,v) = ∇(v) ⊙ (a * ∇(u))
bfe(b::Function,u,v) = b * v * (∇⋅(u))
ffe(f::Function,v) = v ⋅ f
hfe(h::Function,v) = v ⋅ h
aaa(μ::Vector{Float},x) = ∫(∇(v) ⊙ (a * ∇(u)))dΩ

degree=2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γn = BoundaryTriangulation(model, tags=["neumann"])
dΓn = Measure(Γn, degree)

reffe1 = Gridap.ReferenceFE(lagrangian, VectorValue{3,Float}, 2)
reffe2 = Gridap.ReferenceFE(lagrangian, Float, 1; space=:P)

Gμ = ParamFunctional(P,g;FS=Nonaffine(),S=true)
myV = MyTests(model, reffe1; conformity=:H1, dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)
myQ = MyTests(model, reffe2; conformity=:L2)
myP = MyTrials(myQ)

######################### this is ok ####################################
#= V = TestFESpace(model, reffe1; conformity=:H1, dirichlet_tags=["dirichlet"])
U = TrialFESpace(V, g(μ))
P = TestFESpace(model, reffe2; conformity=:L2)
Q = TrialFESpace(P)
a(u,v) = ∫(∇(v) ⊙ (a(μ) * ∇(u)))dΩ
b(u,q) = ∫(q*(∇⋅u))dΩ
rhs(v) = ∫(v ⋅ f(μ))dΩ + ∫(v ⋅ h(μ))dΓn
A = assemble_matrix(a,U,V)
B = assemble_matrix(b,V,Q)
RHS = assemble_vector(rhs,V)
gd = interpolate_dirichlet(g(μ), U)
la(v) = ∫(∇(v) ⊙ (a(μ) * ∇(gd)))dΩ
lb(q) = ∫(q*(∇⋅gd))dΩ
LA = assemble_vector(la,V)
LB = assemble_vector(lb,Q) =#
######################### this is ok ####################################

id = [:A,:B,:F,:H]
FS = [Nonaffine(),Affine(),Nonaffine(),Nonaffine()]
param_fun = [a,b,f,h]
fe_fun = [afe,bfe,ffe,hfe]
fe_spaces = [(myU,myV),(myU,myQ),(myV,),(myV,)]
measure = [dΩ,dΩ,dΩ,dΓn]

I=true
S=true

dict = Dict(:id=>id,:FS=>FS,:param_fun=>param_fun,
  :fe_fun=>fe_fun,:param_space=>[P],:fe_spaces=>fe_spaces,:measure=>measure)
stokes_problem = ParamFEProblem(dict;I,S)
op = param_operator(stokes_problem)

μ = realization(P)
Aapp, LAapp = stokes_problem.param_fe_array[1].array(μ)
Bapp = stokes_problem.param_fe_array[2].array(μ)
Fapp = stokes_problem.param_fe_array[3].array(μ)
Happ = stokes_problem.param_fe_array[4].array(μ)
