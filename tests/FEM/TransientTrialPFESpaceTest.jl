module TransientTrialParamFESpaceTest
using Test
using Gridap.Arrays
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.FESpaces
using Gridap.CellData
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamODEs

function slow_idx(kst::Int,ns::Int)
  Int(floor((kst-1)/ns)+1)
end

function fast_idx(kst::Int,ns::Int)
  ks = mod(kst,ns)
  ks == 0 ? ns : ks
end

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")

g(x,μ,t) = exp(-sum(x)*t/sum(μ))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = 𝑓ₚₜ(g,μ,t)

params = [rand(3),rand(3),rand(3)]
μ = ParamRealization(params)
t = 0:3
μt = TransientParamRealization(μ,t)
Uμt = TransientTrialParamFESpace(V,gμt)
U = Uμt(μt)
dirichlet_values = get_dirichlet_dof_values(U)

t′ = 1:3
@test param_length(U) == length(μt) == length(gμt(μ,t′)) == 9

for i in param_eachindex(U)
  Ũi = param_getindex(U,i)
  Ui = TrialFESpace(V,g(params[fast_idx(i,3)],t′[slow_idx(i,3)]))
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ui)
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ũi)
end
end
