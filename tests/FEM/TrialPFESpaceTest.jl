module TrialParamFESpaceTest
using Test
using Gridap.Arrays
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.FESpaces
using Gridap.CellData
using Mabla.FEM

domain = (0,1,0,1,0,1)
partition = (3,3,3)
model = CartesianDiscreteModel(domain,partition)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe,dirichlet_tags=["tag_01","tag_10"])

g(x,μ) = exp(-sum(x)/sum(μ))
g(μ) = x->g(x,μ)

params = [rand(3),rand(3),rand(3)]
μ = ParamRealization(params)
gμ = 𝑓ₚ(g,μ)
U = TrialParamFESpace(V,gμ)
dirichlet_values = get_dirichlet_dof_values(U)

@test length_dirichlet_values(U) == length(μ) == 3

for i in 1:length_dirichlet_values(U)
  Ũi = FEM.param_getindex(U,i)
  test_single_field_fe_space(Ũi)
  Ui = TrialFESpace(V,g(params[i]))
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ui)
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ũi)
end
end
