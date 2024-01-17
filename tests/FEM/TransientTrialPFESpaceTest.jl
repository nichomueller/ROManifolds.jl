module TransientTrialPFESpaceTest
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

g(x,μ,t) = exp(-sum(x)*t/sum(μ))
g(μ,t) = x->g(x,μ,t)

params = [rand(3),rand(3),rand(3)]
μ = PRealization(params)
t = 1:3
μt = TransientPRealization(μ,t)
gμt = 𝑓ₚ(g,μ,t)
Uμt = TransientTrialPFESpace(V,gμt)
U = Uμt(μt)
dirichlet_values = get_dirichlet_dof_values(U)

@test length_dirichlet_values(U) == length(μt) == length(gμt) == 9

for Ũi in U
  test_single_field_fe_space(Ũi)
  Ui = TrialFESpace(V,γ)
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ui)
  @test dirichlet_values[i] == get_dirichlet_dof_values(Ũi)
end
end
