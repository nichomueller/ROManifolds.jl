# module MultiFieldPFESpacesTests

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

μ = PRealization([[1],[2],[3]])
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

U = TrialPFESpace(V,gμ)
P = TrialPFESpace(Q)

multi_field_style = ConsecutiveMultiFieldStyle()

Y = MultiFieldPFESpace([V,Q],style=multi_field_style)
X = MultiFieldPFESpace([U,P],style=multi_field_style)

@test isa(Y,MultiFieldFESpace)
@test isa(X,MultiFieldPFESpace)
@test isa(X.spaces,Vector{<:SingleFieldPFESpace})
@test get_vector_type(X) <: PArray

@test num_free_dofs(X) == num_free_dofs(U) + num_free_dofs(P)
@test num_free_dofs(X) == num_free_dofs(Y)
@test length(X) == 2
@test typeof(zero_free_values(X)) <: PArray{Vector{Float64},1,Vector{Vector{Float64}},3}

dy = get_fe_basis(Y)
dv, dq = dy

dx = get_trial_fe_basis(X)
du, dp = dx

cellmat = integrate(gμ*dv*du,quad)
cellvec = integrate(gμ*dv*2,quad)
cellmatvec = pair_arrays(cellmat,cellvec)
@test isa(cellmat[end],ArrayBlock{<:PArray})
@test cellmat[1][1,1] != nothing
@test cellmat[1][1,2] == nothing
@test isa(cellvec[end], ArrayBlock{<:PArray})
@test cellvec[1][1] != nothing
@test cellvec[1][2] == nothing

free_values = allocate_parray(rand(num_free_dofs(X)),length(gμ))
xh = FEFunction(X,free_values)

uh,ph = xh

###########
f = uh
trian = get_triangulation(f)
free_values = get_free_dof_values(f)
fe_space = get_fe_space(f)
cell_values = get_cell_dof_values(f,trian)
dirichlet_values = f.dirichlet_values
i = 1
# for i in 1:length_dirichlet_values(fe_space)
  fe_space_i = FEM._getindex(fe_space,i)
  fi = FEFunction(fe_space_i,free_values[i])
  test_fe_function(fi)
  @test free_values[i] == get_free_dof_values(fi)
  @test cell_values[i] == get_cell_dof_values(fi,trian)
  @test dirichlet_values[i] == fi.dirichlet_values
# end
###########

test_fe_function(ph)
map(test_fe_function,xh.single_fe_functions)

@test isa(xh,FEPFunction)
uh, ph = xh
@test isa(uh,FEPFunction)
@test isa(ph,FEPFunction)

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

f(x) = sin(4*pi*(x[1]-x[2]^2))+1
fh = interpolate([f,f],X)
fh = interpolate_everywhere([f,f],X)
fh = interpolate_dirichlet([f,f],X)

# end # module
