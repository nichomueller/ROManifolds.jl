
using Test
using LinearAlgebra
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.Fields
using Gridap.Geometry
using Gridap.TensorValues
using Gridap.CellData
using Gridap.ReferenceFEs
using ForwardDiff
using SparseArrays
using Mabla.FEM

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

μ = ParamRealization([[1],[2],[3]])
f(x,μ) = sum(μ)*x[1]
f(μ) = x -> f(x,μ)
fμ = 𝑓ₚ(f,μ)

V = FESpace(model,ReferenceFE(lagrangian,Float64,2),conformity=:H1)
U = TrialParamFESpace(V,fμ)

dv = get_fe_basis(V)
du = get_trial_fe_basis(U)
uh = FEFunction(U,array_of_similar_arrays(rand(num_free_dofs(U)),length(fμ)))

ener(uh) = ∫( fμ*∇(uh)⋅∇(uh)*0.5 )*dΩ
res(uh) = ∫(fμ*∇(uh)⋅∇(dv))*dΩ
jac(uh) = ∫(fμ*∇(du)⋅∇(dv))*dΩ

cell_r = get_array(res(uh))
cell_j = get_array(jac(uh))
cell_h = cell_j

cell_r_auto = get_array(gradient(ener,uh))
cell_j_auto = get_array(jacobian(res,uh))
cell_h_auto = get_array(hessian(ener,uh))

test_array(cell_r_auto,cell_r,≈)
test_array(cell_j_auto,cell_j,≈)
test_array(cell_h_auto,cell_h,≈)

Γ = BoundaryTriangulation(model)
dΓ = Measure(Γ,2)

ener(uh) = ∫( fμ*∇(uh)⋅∇(uh)*0.5 )*dΓ
res(uh) = ∫( fμ*∇(uh)⋅∇(dv) )*dΓ
jac(uh) = ∫( fμ*∇(du)⋅∇(dv) )*dΓ

cell_r = get_array(res(uh))
cell_j = get_array(jac(uh))
cell_h = cell_j

cell_r_auto = get_array(gradient(ener,uh))
cell_j_auto = get_array(jacobian(res,uh))
cell_h_auto = get_array(hessian(ener,uh))

test_array(cell_r_auto,cell_r,≈)
test_array(cell_j_auto,cell_j,≈)
test_array(cell_h_auto,cell_h,≈)

ener(uh) = ∫( fμ*∇(uh)⋅∇(uh)*0.5 )*dΓ + ∫( fμ*∇(uh)⋅∇(uh)*0.5 )*dΩ
res(uh) = ∫( fμ*∇(uh)⋅∇(dv) )*dΓ + ∫(fμ*∇(uh)⋅∇(dv))*dΩ
jac(uh) = ∫( fμ*∇(du)⋅∇(dv) )*dΓ + ∫(fμ*∇(du)⋅∇(dv))*dΩ

cell_r = res(uh)
cell_j = jac(uh)
cell_h = cell_j

cell_r_auto = gradient(ener,uh)
cell_j_auto = jacobian(res,uh)
cell_h_auto = hessian(ener,uh)

test_array(cell_r_auto[Ω],cell_r[Ω],≈)
test_array(cell_j_auto[Ω],cell_j[Ω],≈)
test_array(cell_h_auto[Ω],cell_h[Ω],≈)

test_array(cell_r_auto[Γ],cell_r[Γ],≈)
test_array(cell_j_auto[Γ],cell_j[Γ],≈)
test_array(cell_h_auto[Γ],cell_h[Γ],≈)
