module MultiFieldParamFESpacesTests

using FillArrays
using Gridap.Arrays
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.CellData
using Gridap.MultiField
using Mabla.FEM
using Test

μ = ParamRealization([[1],[2],[3]])
g(x,μ) = 1+sum(μ)
g(μ) = x -> g(x,μ)
gμ = 𝑓ₚ(g,μ)

order = 2

domain = (0,1,0,1)
partition = (3,3)
model = CartesianDiscreteModel(domain,partition)

trian = get_triangulation(model)
degree = order
quad = CellQuadrature(trian,degree)

V = TestFESpace(model,ReferenceFE(lagrangian,Float64,order);conformity=:H1)
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,order-1),conformity=:L2)

U = TrialParamFESpace(V,gμ)
P = TrialParamFESpace(Q)

multi_field_style = ConsecutiveMultiFieldStyle()

Y = MultiFieldParamFESpace([V,Q],style=multi_field_style)
X = MultiFieldParamFESpace([U,P],style=multi_field_style)

@test isa(Y,MultiFieldFESpace)
@test isa(X,MultiFieldParamFESpace)
@test isa(X.spaces,Vector{<:SingleFieldParamFESpace})
@test get_vector_type(X) <: ParamArray

@test num_free_dofs(X) == num_free_dofs(U) + num_free_dofs(P)
@test num_free_dofs(X) == num_free_dofs(Y)
@test length(X) == 2
@test typeof(zero_free_values(X)) <: ParamArray{Vector{Float64},1,3,Vector{Vector{Float64}}}

dy = get_fe_basis(Y)
dv, dq = dy

dx = get_trial_fe_basis(X)
du, dp = dx

cellmat = integrate(gμ*dv*du,quad)
cellvec = integrate(gμ*dv*2,quad)
cellmatvec = pair_arrays(cellmat,cellvec)
@test isa(cellmat[end],ArrayBlock{<:ParamArray})
@test cellmat[1][1,1] != nothing
@test cellmat[1][1,2] == nothing
@test isa(cellvec[end], ArrayBlock{<:ParamArray})
@test cellvec[1][1] != nothing
@test cellvec[1][2] == nothing

free_values = array_of_similar_arrays(rand(num_free_dofs(X)),length(gμ))
xh = FEFunction(X,free_values)
test_fe_function(xh)
uh,ph = xh
@test isa(xh,FEFunction)
@test isa(uh,FEFunction)
@test isa(ph,FEFunction)

cell_isconstr = get_cell_isconstrained(X,trian)
@test cell_isconstr == Fill(false,num_cells(model))

cell_constr = get_cell_constraints(X,trian)
@test isa(cell_constr,LazyArray{<:Fill{<:BlockMap}})

cell_dof_ids = get_cell_dof_ids(X,trian)
@test isa(cell_dof_ids,LazyArray{<:Fill{<:BlockMap}})

cf = CellField(X,get_cell_dof_ids(X,trian))
@test isa(cf,MultiFieldCellField)

test_fe_space(X,cellmatvec,cellmat,cellvec,trian)
test_fe_space(Y,cellmatvec,cellmat,cellvec,trian)

#using Gridap.Visualization
#writevtk(trian,"trian";nsubcells=30,cellfields=["uh" => uh, "ph"=> ph])

fh = interpolate([gμ,gμ],X)
fh = interpolate_everywhere([gμ,gμ],X)
fh = interpolate_dirichlet([gμ,gμ],X)

end # module
