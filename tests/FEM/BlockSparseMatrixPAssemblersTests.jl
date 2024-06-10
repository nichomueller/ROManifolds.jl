# module BlockMatrixAssemblersTests
using Test, BlockArrays, SparseArrays, LinearAlgebra

using Gridap
using Gridap.Algebra, Gridap.CellData, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces

import ArraysOfArrays: innersize
############################################################################################
parametric = true
μ = ParamRealization([[1],[2],[3]])
sol(x,μ) = (1+sum(μ))*sum(x)
sol(μ) = x -> sol(x,μ)
solμ = 𝑓ₚ(sol,μ)

f(x) = sum(x)

model = CartesianDiscreteModel((0.0,1.0,0.0,1.0),(5,5))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω,reffe;dirichlet_tags="boundary")

if parametric
  U = TrialParamFESpace(V,solμ)
  biform((u1,u2),(v1,v2)) = ∫(solμ*∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
  liform((v1,v2)) = ∫(solμ*v1 - v2)*dΩ
else
  U = TrialFESpace(V,f)
  biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
  liform((v1,v2)) = ∫(v1 - v2)*dΩ
end

############################################################################################
# Normal assembly

Y = MultiFieldParamFESpace(fill(V,2))
X = MultiFieldParamFESpace(fill(U,2))

u = get_trial_fe_basis(X)
v = get_fe_basis(Y)

data = collect_cell_matrix_and_vector(X,Y,biform(u,v),liform(v))
matdata = collect_cell_matrix(X,Y,biform(u,v))
vecdata = collect_cell_vector(Y,liform(v))

assem = SparseMatrixAssembler(X,Y)
if parametric
  assem = get_param_assembler(assem,μ)
end
A1 = assemble_matrix(assem,matdata)
b1 = assemble_vector(assem,vecdata)
A2,b2 = assemble_matrix_and_vector(assem,data)

############################################################################################
# Block MultiFieldStyle

function my_test_fe_space(f::FESpace,cell_matvec,cell_mat,cell_vec,trian)
  my_test_fe_space(f)

  cm = attach_constraints_cols(f,cell_mat,trian)
  if ! has_constraints(f)
    @test cm === cell_mat
  end
  cm = attach_constraints_rows(f,cell_mat,trian)
  if ! has_constraints(f)
    @test cm === cell_mat
  end

  cv = attach_constraints_rows(f,cell_vec,trian)
  if ! has_constraints(f)
    @test cv === cell_vec
  end

  cmv = attach_constraints_cols(f,cell_matvec,trian)
  if ! has_constraints(f)
    @test cmv === cell_matvec
  end
  cmv = attach_constraints_rows(f,cell_matvec,trian)
  if ! has_constraints(f)
    @test cmv === cell_matvec
  end

end

function my_test_fe_space(f::FESpace)
  trian = get_triangulation(f)
  @test isa(trian,Triangulation)
  free_values = zero_free_values(f)
  @test sum(map(sum,innersize(free_values))) == num_free_dofs(f)
  V = get_vector_type(f)
  @test typeof(free_values) == V
  fe_function = FEFunction(f,free_values)
  test_fe_function(fe_function)
  fe_basis = get_fe_basis(f)
  @test isa(has_constraints(f),Bool)
  @test isa(has_constraints(typeof(f)),Bool)
  @test length(get_cell_dof_ids(f,trian)) == num_cells(fe_basis)
  @test length(get_cell_constraints(f,trian)) == num_cells(fe_basis)
  @test length(get_cell_isconstrained(f,trian)) == num_cells(fe_basis)
  @test CellField(f,get_cell_dof_ids(f,trian)) != nothing
end

Yb = MultiFieldParamFESpace(fill(V,2);style=BlockMultiFieldStyle())
Xb = MultiFieldParamFESpace(fill(U,2);style=BlockMultiFieldStyle())
test_fe_space(Yb)
my_test_fe_space(Xb)

ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)

bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))
my_test_fe_space(Xb,bdata[1][1][1],bmatdata[1][1],bvecdata[1][1],Ω)
test_fe_space(Yb,bdata[1][1][1],bmatdata[1][1],bvecdata[1][1],Ω)

############################################################################################
# Block Assembly

assem_blocks = SparseMatrixAssembler(Xb,Yb)
if parametric
  assem_blocks = get_param_assembler(assem_blocks,μ)
end
ParamFESpaces.test_passembler(assem_blocks,bmatdata,bvecdata,bdata)

A1_blocks = assemble_matrix(assem_blocks,bmatdata)
b1_blocks = assemble_vector(assem_blocks,bvecdata)
for i = 1:length(solμ)
  @test A1[i] ≈ A1_blocks[i]
  @test b1[i] ≈ b1_blocks[i]
end

y1_blocks = similar(b1_blocks)
mul!(y1_blocks,A1_blocks,b1_blocks)
y1 = similar(b1)
mul!(y1,A1,b1)
for i = 1:length(solμ)
  @test y1[i] ≈ y1_blocks[i]
end

A2_blocks, b2_blocks = assemble_matrix_and_vector(assem_blocks,bdata)
for i = 1:length(solμ)
  @test A2[i] ≈ A2_blocks[i]
  @test b2[i] ≈ b2_blocks[i]
end

A3_blocks = allocate_matrix(assem_blocks,bmatdata)
b3_blocks = allocate_vector(assem_blocks,bvecdata)
assemble_matrix!(A3_blocks,assem_blocks,bmatdata)
assemble_vector!(b3_blocks,assem_blocks,bvecdata)
for i = 1:length(solμ)
  @test A1[i] ≈ A3_blocks[i]
  @test b1_blocks[i] ≈ b3_blocks[i]
end

A4_blocks, b4_blocks = allocate_matrix_and_vector(assem_blocks,bdata)
assemble_matrix_and_vector!(A4_blocks,b4_blocks,assem_blocks,bdata)
for i = 1:length(solμ)
  @test A4_blocks[i] ≈ A2_blocks[i]
  @test b4_blocks[i] ≈ b2_blocks[i]
end

############################################################################################

# op = AffineFEOperator(biform,liform,X,Y)
# block_op = AffineFEOperator(biform,liform,Xb,Yb)

# @test get_matrix(op) ≈ get_matrix(block_op)
# @test get_vector(op) ≈ get_vector(block_op)

# end # module

############

A = allocate_matrix(assem_blocks,bmatdata)
@test FESpaces.num_cols(assem_blocks) == size(A,2)
@test FESpaces.num_rows(assem_blocks) == size(A,1)
assemble_matrix!(A,assem_blocks,matdata)

aa = assem_blocks
matdata = bmatdata

A = allocate_matrix(aa,matdata)
@test FESpaces.num_cols(aa) == size(A,2)
@test FESpaces.num_rows(aa) == size(A,1)
assemble_matrix!(A,aa,matdata)
assemble_matrix_add!(A,aa,matdata)
A = assemble_matrix(aa,matdata)
@test FESpaces.num_cols(aa) == size(A,2)
@test FESpaces.num_rows(aa) == size(A,1)
b = allocate_vector(aa,vecdata)
@test length(testitem(b)) == FESpaces.num_rows(aa)
assemble_vector!(b,aa,vecdata)
assemble_vector_add!(b,aa,vecdata)
b = assemble_vector(aa,vecdata)
@test length(testitem(b)) == FESpaces.num_rows(aa)
A, b = allocate_matrix_and_vector(aa,data)
assemble_matrix_and_vector!(A,b,aa,data)
assemble_matrix_and_vector_add!(A,b,aa,data)
@test FESpaces.num_cols(aa) == size(A,2)
@test FESpaces.num_rows(aa) == size(A,1)
@test length(testitem(b)) == num_rows(aa)
A, b = assemble_matrix_and_vector(aa,data)
@test FESpaces.num_cols(aa) == size(A,2)
@test FESpaces.num_rows(aa) == size(A,1)
@test length(testitem(b)) == FESpaces.num_rows(aa)


m1 = nz_counter(get_matrix_builder(aa),(get_rows(aa),get_cols(aa)))
symbolic_loop_matrix!(m1,aa,matdata)
m2 = nz_allocation(m1)
symbolic_loop_matrix!(m2,aa,matdata)
m3 = create_from_nz(m2)

array = map(Algebra.create_from_nz,m2.array)
mortar(array)
