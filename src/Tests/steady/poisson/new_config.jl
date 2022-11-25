include("../../../src/Utils/Utils.jl")
include("../../../src/FEM/FEM.jl")
include("../../../src/FEM/NewFESpaces.jl")
include("../../../src/FEM/NewTypes.jl")
include("../../../src/FEM/NewFEOperators.jl")
include("../../../src/FEM/ParamOperatorInterfaces.jl")
include("../../../src/FEM/OperatorsFromProblem.jl")

root = "/home/nicholasmueller/git_repos/Mabla.jl"
mesh_name = "cube15x15x15.json"
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
f(μ::Vector{Float},x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
f(μ::Vector{Float}) = x->f(μ,x)
h(μ::Vector{Float},x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
h(μ::Vector{Float}) = x->h(μ,x)
g(μ::Vector{Float},x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
g(μ::Vector{Float}) = x->g(μ,x)

afe(a::Function,u,v) = ∇(v) ⋅ (a * ∇(u))
ffe(f::Function,v) = v * f
hfe(h::Function,v) = v * h

degree=1
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γn = BoundaryTriangulation(model, tags=["neumann"])
dΓn = Measure(Γn, degree)

reffe = Gridap.ReferenceFE(lagrangian, Float, 2)

Gμ = ParamFunctional(P,g;FS=Nonaffine(),S=true)
myV = MyTests(model, reffe; conformity=:H1, dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)

#= ######################### this is ok ####################################
V = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["dirichlet"])
U = TrialFESpace(V, g(μ))
a(u, v) = ∫(∇(v) ⋅ (α(μ) * ∇(u)))dΩ
rhs(v) = ∫(v * f(μ))dΩ + ∫(v * h(μ))dΓn
A = assemble_matrix(a,U,V)
RHS = assemble_vector(rhs,V)
gd = interpolate_dirichlet(g(μ), U)
la(v) = ∫(∇(v) ⋅ (α(μ) * ∇(gd)))dΩ
LA = assemble_vector(la,V) =#

id = [:A,:F,:H]
FS = [Nonaffine(),Nonaffine(),Nonaffine()]
param_fun = [a,f,h]
fe_fun = [afe,ffe,hfe]
fe_spaces = [(myU,myV),(myV,),(myV,)]
measure = [dΩ,dΩ,dΓn]

I=false
S=true

dict = Dict(:id=>id,:FS=>FS,:param_fun=>param_fun,
  :fe_fun=>fe_fun,:param_space=>[P],:fe_spaces=>fe_spaces,:measure=>measure)
poisson_problem = ParamFEProblem(dict;I,S)
op = param_operator(poisson_problem)
#ufield = solve(LinearFESolver(),op)

μ = realization(P)
poisson_problem(μ)



#operator = ParamAffineFEOperator(a, rhs, P, U, V)

#= snapshots::SnapshotType = compute_snaptshots(A::ParamOperator)
save_to_disk(snaptshots,"my_name.xxx")

new_snapshots = upload_from_disk("my_name.xxx")

struct Snapshots
  op::ParamOperator
  A::AbstractArray
end

# API
function save_to_disk(s::Snapshots)
  s_a = get_array(s)
  # print to file
end

function get_array(s::Snapshots)
  # compute the array
end

function upload_from_disk(file)
  # read the array from disk
end

mutable struct Snapshots{O,T} where {O<:ParamOperator,T}
  op::O
  A::AbstractArray{T}
end
=#
