module SteadyHyperElasticityPOD

using ROM
using DrWatson

using Gridap
import Gridap.FESpaces: NonlinearFESolver

using GridapSolvers
import GridapSolvers.NonlinearSolvers: NewtonSolver

include("ExamplesInterface.jl")

pranges = (1e2,5*1e2,0.25,1.25,1e-1,1.0)
pspace = ParamSpace(pranges)

domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",[1,3,7])
add_tag_from_tags!(labels,"dirichlet",[2,4,8])

Ω = Triangulation(model)

λ(μ) = μ[1]
p(μ) = μ[2]

# Deformation Gradient
F(∇u) = one(∇u) + ∇u'
J(F) = sqrt(det(C(F)))
dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

# Right Cauchy-green deformation tensor
C(F) = (F')⋅F

# Constitutive law (Neo hookean)
function S(μ)
  function _S(∇u)
    Cinv = inv(C(F(∇u)))
    p(μ)*(one(∇u)-Cinv) + λ(μ)*log(J(F(∇u)))*Cinv
  end
  return _S
end
Sμ(μ) = ParamFunction(S,μ)

function dS(μ)
  function _dS((∇du,∇u))
    Cinv = inv(C(F(∇u)))
    _dE = dE(∇du,∇u)
    λ(μ)*(Cinv⊙_dE)*Cinv + 2*(p(μ)-λ(μ)*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
  end
  return _dS
end
dSμ(μ) = ParamFunction(dS,μ)

g(μ) = x -> VectorValue(μ[3],0.0)
gμ(μ) = ParamFunction(g,μ)

g0(μ) = x -> VectorValue(0.0,0.0)
g0μ(μ) = ParamFunction(g0,μ)

order = 1
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
trial = ParamTrialFESpace(test,[g0μ,gμ])

degree = 2*order
dΩ = Measure(Ω,degree)

res(μ,u,v,dΩ) = ∫( (dE∘(∇(v),∇(u))) ⊙ (Sμ(μ)∘∇(u)) )dΩ
jac(μ,u,du,v,dΩ) = (
  ∫( (dE∘(∇(v),∇(u))) ⊙ (dSμ(μ)∘((∇(du),∇(u)))) )dΩ +
  ∫( ∇(v) ⊙ ( (Sμ(μ)∘∇(u))⋅∇(du) ) )dΩ
  )

trian_res = (Ω,)
trian_jac = (Ω,)
domains = FEDomains(trian_res,trian_jac)

feop = ParamFEOperator(res,jac,pspace,trial,test,domains)

fesolver = NonlinearFESolver(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true))

tol = 1e-5
energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ
state_reduction = Reduction(tol,energy;nparams=10)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=10,nparams_jac=10)

dir = datadir("hyper_elasticity_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
ExamplesInterface.run_test(dir,rbsolver,feop,tols)

end

using Gridap
using GridapSolvers
import GridapSolvers.NonlinearSolvers: NewtonSolver

const λ = 100.0
const μ = 1.0

F(∇u) = one(∇u) + ∇u'

J(F) = sqrt(det(C(F)))

#Green strain

dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

C(F) = (F')⋅F

function S(∇u)
  Cinv = inv(C(F(∇u)))
  μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
end

function dS(∇du,∇u)
  Cinv = inv(C(F(∇u)))
  _dE = dE(∇du,∇u)
  λ*(Cinv⊙_dE)*Cinv + 2*(μ-λ*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
end

σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'

domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri_0",[1,3,7])
add_tag_from_tags!(labels,"diri_1",[2,4,8])

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

res(u,v) = ∫( (dE∘(∇(v),∇(u))) ⊙ (S∘∇(u)) )*dΩ

jac_mat(u,du,v) =  ∫( (dE∘(∇(v),∇(u))) ⊙ (dS∘(∇(du),∇(u))) )*dΩ

jac_geo(u,du,v) = ∫( ∇(v) ⊙ ( (S∘∇(u))⋅∇(du) ) )*dΩ

jac(u,du,v) = jac_mat(u,du,v) + jac_geo(u,du,v)

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},1)
V = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags = ["diri_0", "diri_1"])

solver = FESolver(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true))

g0 = VectorValue(0.0,0.0)
g1 = VectorValue(-0.5,0.0)
U = TrialFESpace(V,[g0,g1])

op = FEOperator(res,jac,U,V)

uh = solve(solver,op)
