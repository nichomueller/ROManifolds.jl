include("../../../FEM/FEM.jl")

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
      add_tag_from_tags!(labels,tags[i],bnds[i])
    end
  end
end
bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
set_labels!(bnd_info)

degree=2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

ranges = [[1., 10.], [1., 10.], [1., 10.],
          [1., 10.], [1., 10.], [1., 10.]]
param_space = ParamSpace(ranges,UniformSampling())

a(x,t::Real,μ::Vector{Float}) = 1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
a(t::Real,μ::Vector{Float}) = x->a(x,t,μ)
a(μ::Vector{Float}) = t->a(t,μ)
b(x,t::Real,μ::Vector{Float}) = 1.
b(t::Real,μ::Vector{Float}) = x->b(x,t,μ)
b(μ::Vector{Float}) = t->b(t,μ)
f(x,t::Real,μ::Vector{Float}) = 1. + sin(t)*Point(μ[4:6]).*x
f(t::Real,μ::Vector{Float}) = x->f(x,t,μ)
f(μ::Vector{Float}) = t->f(t,μ)
h(x,t::Real,μ::Vector{Float}) = 1. + sin(t)*Point(μ[4:6]).*x
h(t::Real,μ::Vector{Float}) = x->h(x,t,μ)
h(μ::Vector{Float}) = t->h(t,μ)
g(x,t::Real,μ::Vector{Float}) = 1. + sin(t)*Point(μ[4:6]).*x
g(t::Real,μ::Vector{Float}) = x->g(x,t,μ)
g(μ::Vector{Float}) = t->g(t,μ)

mfe(μ,t,u,v) = ∫(v⋅u)dΩ
afe(μ,t,u,v) = ∫(a(t,μ)*∇(v) ⊙ ∇(u))dΩ
bfe(μ,t,u,q) = ∫(b(t,μ)*q*(∇⋅(u)))dΩ
ffe(μ,t,v) = ∫(f(t,μ)⋅v)dΩ
hfe(μ,t,v) = ∫(h(t,μ)⋅v)dΓn
conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
#= c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ =#
c(z,u,v) = ∫(v ⊙ (∇(u)'⋅z))dΩ
d(z,u,v) = ∫(v ⊙ (∇(z)'⋅u))dΩ

rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)#∫(a(t,μ)*∇(v)⊙∇(u) - b(t,μ)*((∇⋅v)*p + q*(∇⋅u)))dΩ

res(μ,t,(u,p),(v,q)) = mfe(μ,t,(∂t(u),∂t(p)),(v,q)) + lhs(μ,t,(u,p),(v,q)) + c(u,u,v) - rhs(μ,t,(v,q))
jac(μ,t,(u,p),(du,dp),(v,q)) = lhs(μ,t,(du,dp),(v,q)) + d(u,du,v)
jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = mfe(μ,t,(dut,dpt),(v,q))

reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)
reffe2 = Gridap.ReferenceFE(lagrangian,Float,degree-1;space=:P)

I=true
S=false

Gμ = ParamFunctional(param_space,g;S)
myV = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)
myQ = MyTests(model,reffe2;conformity=:L2)
myP = MyTrials(myQ)

X = ParamTransientMultiFieldFESpace([myU.trial,myP.trial])
Y = ParamTransientMultiFieldFESpace([myV.test,myQ.test])
op = ParamTransientFEOperator(res,jac,jac_t,param_space,X,Y)
nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
ode_solver = ThetaMethod(nls,0.025,0.5)
ye = solve(ode_solver,op,0.,0.5)
x = Vector{Float}[]
count = 1
for (xₕ, _) in ye[1]
  println("Time step: $count")
  push!(x,get_free_dof_values(xₕ))
  count += 1
end

opA = ParamVarOperator(a,afe,param_space,myU,myV)
opB = ParamVarOperator(b,bfe,param_space,myU,myQ)
opF = ParamVarOperator(f,ffe,param_space,myV)
opH = ParamVarOperator(h,hfe,param_space,myV)

stokes_problem = Problem(μ,[uh,ph],[opA,opB,opF,opH])
