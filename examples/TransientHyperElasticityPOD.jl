module ElasticitySteady

using ROM
using Gridap
using DrWatson

pranges = (1e10,9*1e10,0.25,0.42,-4*1e5,4*1e5,-4*1e5,4*1e5,-4*1e5,4*1e5)
ptspace = ParamSpace(pranges)

domain = (0,1,0,0.5,0,0.25)
partition = (20,10,10)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,5,7,13,15,17,19,25])
add_tag_from_tags!(labels,"neumann1",[22])
add_tag_from_tags!(labels,"neumann2",[24])
add_tag_from_tags!(labels,"neumann3",[26])

Ω = Triangulation(model)
Γ1 = BoundaryTriangulation(model,tags="neumann1")
Γ2 = BoundaryTriangulation(model,tags="neumann2")
Γ3 = BoundaryTriangulation(model,tags="neumann3")

# Deformation Gradient
F(∇u) = one(∇u) + ∇u'
J(F) = sqrt(det(C(F)))
dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

# Right Cauchy-green deformation tensor
C(F) = (F')⋅F

# Constitutive law (Neo hookean)

function S(∇u)
  Cinv = inv(C(F(∇u)))
  μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
end

function dS(∇du,∇u)
  Cinv = inv(C(F(∇u)))
  _dE = dE(∇du,∇u)
  λ*(Cinv⊙_dE)*Cinv + 2*(μ-λ*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
end

λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
p(μ) = μ[1]/(2(1+μ[2]))

σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'

σ(ε,μ) = exp(sin(2*π*t/tf))*(λ(μ)*tr(ε)*one(ε) + 2*p(μ)*ε)
σμ(μ) = ParamFunction(ε -> σ(ε,μ),μ)

h1(x,μ) = VectorValue(0.0,0.0,μ[3])
h1μ(μ) = ParamFunction(x->h1(x,μ),μ)

h2(x,μ) = VectorValue(0.0,μ[4],0.0)
h2μ(μ) = ParamFunction(x->h2(x,μ),μ)

h3(x,μ) = VectorValue(μ[5]*x[1],0.0,0.0)
h3μ(μ) = ParamFunction(x->h3(x,μ),μ)

g(x,μ) = VectorValue(0.0,0.0,0.0)
gμ(μ) = ParamFunction(x->g(x,μ),μ)

order = 2
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TrialParamFESpace(test,gμ)

degree = 2*order
dΩ = Measure(Ω,degree)
dΓ1 = Measure(Γ1,degree)
dΓ2 = Measure(Γ2,degree)
dΓ3 = Measure(Γ3,degree)

stiffness(μ,u,v,dΩ) = ∫( ε(v) ⊙ (σμ(μ)∘ε(u)) )*dΩ
res(μ,u,v,dΩ,dΓ1,dΓ2,dΓ3) = stiffness(μ,u,v,dΩ) - (
  ∫(v⋅h1μ(μ))dΓ1 + ∫(v⋅h2μ(μ))dΓ2 + ∫(v⋅h3μ(μ))dΓ3)

end
