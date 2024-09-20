module MultiFieldCellParamFieldsTests

using FillArrays
using Gridap.Arrays
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.TensorValues
using Mabla.FEM
using Random
using StaticArrays
using Test

μ = Realization([[1],[2],[3]])

domain = (0,1,0,1)
cells = (2,2)
model = CartesianDiscreteModel(domain,cells)

trian = Triangulation(model)

u1(x,μ) = sin(x[1])*(1+sum(μ))
u1(μ) = x -> u1(x,μ)
u1μ = 𝑓ₚ(u1,μ)
cf1 = CellField(u1μ,trian)

u2(x,μ) = cos(x[2])*(1+sum(μ))
u2(μ) = x -> u2(x,μ)
u2μ = 𝑓ₚ(u2,μ)
cf2 = CellField(u2μ,trian)

cf = MultiFieldCellField([cf1,cf2])

@test cf1 === cf[1]
@test cf2 === cf[2]

_cf1, _cf2 = cf

@test cf1 === _cf1
@test cf2 === _cf2

order = 2
degree = order
quad = CellQuadrature(trian,degree)
x = get_cell_points(quad)

trian_Γ = SkeletonTriangulation(model)
quad_Γ = CellQuadrature(trian_Γ,degree)
x_Γ = get_cell_points(quad_Γ)

V = TestFESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},order);conformity=:H1)
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,order-1),conformity=:L2)

U = TrialFESpace(V)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

dv, dq = get_fe_basis(Y)
du, dp = get_trial_fe_basis(X)

n = VectorValue(1,2)

cellmat = integrate( u1μ*(n⋅dv)*dp + dq*dp, quad)
cellvec = integrate( u1μ*(n⋅dv), quad)
@test isa(cellvec[end],ArrayBlock{<:AbstractParamArray})
@test isa(cellmat[end],ArrayBlock{<:AbstractParamArray})

cellmat1 = integrate( u1μ*((n⋅dv) - dq)*((n⋅du) + dp), quad)
cellmat2 = integrate( u1μ*(n⋅dv)*(n⋅du) + u1μ*(n⋅dv)*dp - u1μ*dq*(n⋅du) - u1μ*dq*dp, quad)
test_array(cellmat1,cellmat2,≈)

cellmat1 = integrate( u1μ*(n⋅dv)*2, quad)
cellmat2 = integrate( u1μ*(n⋅dv)*fill(2,num_cells(trian)), quad)
test_array(cellmat1,cellmat2,≈)

α = CellField(u1μ,trian)
op2(u,∇u,α) = α*(∇u⋅u)
cellmat1 = integrate( dv⋅(op2∘(du,∇(du),α)),quad)
cellmat2 = integrate( dv⋅(α*(∇(du)⋅du)),quad)
test_array(cellmat1,cellmat2,≈)

conv(u,∇u,α) = α*(u⋅∇u)
dconv(du,∇du,u,∇u,α) = conv(u,∇du,α)+conv(du,∇u,α)

u = zero(U)
cellvec2 = integrate(dv⊙(α*(u⋅∇(u))),quad)
cellvec1 = integrate(dv⊙(conv∘(u,∇(u),α)),quad)
test_array(cellvec1,cellvec2,≈)

end # module
