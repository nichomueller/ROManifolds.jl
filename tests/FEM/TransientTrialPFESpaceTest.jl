module TransientTrialParamFESpaceTest
using Test
using Gridap.Arrays
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.FESpaces
using Gridap.CellData
using Mabla.FEM

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
t = 1:3
μt = TransientParamRealization(μ,t)
Uμt = TransientTrialParamFESpace(V,gμt)
U = Uμt(μt)
dirichlet_values = get_dirichlet_dof_values(U)

@test length_dirichlet_values(U) == length(μt) == length(gμt(μ,t)) == 9

for i in 1:length_dirichlet_values(U)
  Ũi = FEM.param_getindex(U,i)
  Ui = TrialFESpace(V,g(params[slow_idx(i,3)],t[fast_idx(i,3)]))
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ui)
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ũi)
end
end
