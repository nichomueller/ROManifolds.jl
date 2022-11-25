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
param_space = ParamSpace(ranges,UniformSampling())

a(x,t::Real,μ::Vector{Float}) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
a(t::Real,μ::Vector{Float}) = x->a(x,t,μ)
a(μ::Vector{Float}) = t->a(t,μ)
f(x,t::Real,μ::Vector{Float}) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
f(t::Real,μ::Vector{Float}) = x->f(x,t,μ)
f(μ::Vector{Float}) = t->f(t,μ)
h(x,t::Real,μ::Vector{Float}) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
h(t::Real,μ::Vector{Float}) = x->h(x,t,μ)
h(μ::Vector{Float}) = t->h(t,μ)
g(x,t::Real,μ::Vector{Float}) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
g(t::Real,μ::Vector{Float}) = x->g(x,t,μ)
g(μ::Vector{Float}) = t->g(t,μ)

mfe(μ,t,u,v) = ∫(v * u)dΩ
afe(μ,t,u,v) = ∫(a(t,μ) * ∇(v) ⋅ ∇(u))dΩ
ffe(μ,t,v) = ∫(f(t,μ) * v)dΩ
hfe(μ,t,v) = ∫(h(t,μ) * v)dΓn

rhs(μ,t,v) = ffe(μ,t,v) + hfe(μ,t,v)

reffe = Gridap.ReferenceFE(lagrangian,Float,1)

I=false
S=false

Gμ = ParamFunctional(param_space,g;S)
myV = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)

op = ParamTransientAffineFEOperator(mfe,afe,rhs,param_space,myU.trial,myV.test)
ode_solver = ThetaMethod(LUSolver(),0.025,0.5)
ye = solve(ode_solver,op,0.,0.5)
count = 1
for (xₕ, _) in ye[1]
  println("Time step: $count")
  println(get_free_dof_values(xₕ))
  count += 1
end

opA = ParamVarOperator(a,afe,param_space,myU,myV)
opB = ParamVarOperator(b,bfe,param_space,myU,myQ)
opF = ParamVarOperator(f,ffe,param_space,myV)
opH = ParamVarOperator(h,hfe,param_space,myV)

stokes_problem = Problem(μ,[uh,ph],[opA,opB,opF,opH])
