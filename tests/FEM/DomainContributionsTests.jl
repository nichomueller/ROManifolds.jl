using Test
using FillArrays
using Gridap.Helpers
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.CellData
using Gridap.TensorValues
using Gridap.Geometry
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces

lazy_getter(a,i=1) = lazy_map(x->param_getindex(x,i),a)

domain = (0,1,0,1)
cells = (2,2)
model = simplexify(CartesianDiscreteModel(domain,cells))

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
Λ = SkeletonTriangulation(model)
n_Λ = get_normal_vector(Λ)

degree = 2
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΛ = Measure(Λ,degree)

v = GenericCellField(get_cell_shapefuns(Ω),Ω,ReferenceDomain())
u = GenericCellField(lazy_map(transpose,get_data(v)),Ω,ReferenceDomain())

μ = Realization([[1],[2],[3]])
μ₀ = Realization([[0],[0],[0]])
f(x,μ) = 1+sum(μ)
f(μ) = x -> f(x,μ)
fμ = 𝑓ₚ(f,μ)
fμ₀ = 𝑓ₚ(f,μ₀)

a = ∫(fμ*u*v)*dΩ + ∫(fμ*u*v)*dΓ + ∫(fμ*∇(u)⋅∇(v))*dΩ
@test num_domains(a) == 2
@test Ω in get_domains(a)
@test Γ in get_domains(a)
@test isa(get_contribution(a,Ω),LazyArray{<:Any,<:ParamArray})
@test length(lazy_getter(get_contribution(a,Ω))) == num_cells(Ω)
@test param_length(first(get_contribution(a,Ω))) == 3
b = ∫(u*v)*dΩ + ∫(u*v)*dΓ + ∫(∇(u)⋅∇(v))*dΩ
@test 2*sum(b) == sum(a)[1]

a = ∫(fμ₀)*dΩ + ∫(fμ₀)*dΓ
@test all(sum(a) .≈ 5)
@test all(sum(2*a) .≈ 10)
@test all(sum(a*2) .≈ 10)

f1(x,μ) = 2*x[1]*(1+sum(μ))
f1(μ) = x -> f1(x,μ)
f1μ = 𝑓ₚ(f1,μ₀)
f2(x,μ) = 3*x[2]*(1+sum(μ))
f2(μ) = x -> f2(x,μ)
f2μ = 𝑓ₚ(f2,μ₀)
u = CellField(f1μ,Ω)
v = CellField(f2μ,Ω)

a = ∫(jump(u))*dΛ
@test all(sum(a) .+ 1 .≈ 1)

a = ∫( (n_Λ.⁺⋅∇(v.⁻))*jump(n_Λ⋅∇(u)) )*dΛ
@test all(sum(a) .+ 1 .≈ 1)

quad = Quadrature(duffy,2)
dΩ = Measure(Ω,quad)
s = ∫(fμ₀)dΩ
@test all(sum(s) .≈ 1)

dΩ = Measure(Ω,degree,T=Float32)
dΓ = Measure(Γ,degree,T=Float32)
dΛ = Measure(Λ,degree,T=Float32)

a = ∫(fμ₀)*dΩ + ∫(fμ₀)*dΓ
@test all(isapprox.(sum(a),5,atol=1e-6))
@test all(isapprox.(sum(2*a),10,atol=1e-6))
@test all(isapprox.(sum(a*2),10,atol=1e-6))

u = CellField(f1μ,Ω)
v = CellField(f2μ,Ω)

a = ∫(jump(u))*dΛ
@test all(sum(a) .+ 1 .≈ 1)

a = ∫( (n_Λ.⁺⋅∇(v.⁻))*jump(n_Λ⋅∇(u)) )*dΛ
@test all(sum(a) .+ 1 .≈ 1)

quad = Quadrature(duffy,2,T=Float32)
dΩ = Measure(Ω,quad)
s = ∫(fμ₀)dΩ
@test all(sum(s) .≈ 1)
