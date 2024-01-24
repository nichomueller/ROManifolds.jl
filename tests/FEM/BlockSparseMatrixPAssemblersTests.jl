# module BlockMatrixAssemblersTests
using Test, BlockArrays, SparseArrays, LinearAlgebra

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using Mabla.FEM

############################################################################################
μ = PRealization([[1],[2],[3]])
sol(x,μ) = (1+sum(μ))*sum(x)
sol(μ) = x -> sol(x,μ)
solμ = 𝑓ₚ(sol,μ)

model = CartesianDiscreteModel((0.0,1.0,0.0,1.0),(5,5))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω,reffe;dirichlet_tags="boundary")
U = TrialPFESpace(V,solμ)

dΩ = Measure(Ω, 2)
biform((u1,u2),(v1,v2)) = ∫(solμ*∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
liform((v1,v2)) = ∫(solμ*v1 - v2)*dΩ

############################################################################################
# Normal assembly

Y = MultiFieldPFESpace(fill(V,2))
X = MultiFieldPFESpace(fill(U,2))

u = get_trial_fe_basis(X)
v = get_fe_basis(Y)

data = collect_cell_matrix_and_vector(X,Y,biform(u,v),liform(v))
matdata = collect_cell_matrix(X,Y,biform(u,v))
vecdata = collect_cell_vector(Y,liform(v))

assem = SparseMatrixAssembler(X,Y)
A1 = assemble_matrix(assem,matdata)
b1 = assemble_vector(assem,vecdata)
A2,b2 = assemble_matrix_and_vector(assem,data)

############################################################################################
# Block MultiFieldStyle

Yb = MultiFieldPFESpace(fill(V,2);style=BlockMultiFieldStyle())
Xb = MultiFieldPFESpace(fill(U,2);style=BlockMultiFieldStyle())
test_fe_space(Yb)
test_fe_space(Xb)

ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)

bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))
test_fe_space(Xb,bdata[1][1][1],bmatdata[1][1],bvecdata[1][1],Ω)
test_fe_space(Yb,bdata[1][1][1],bmatdata[1][1],bvecdata[1][1],Ω)

############################################################################################
# Block Assembly

assem_blocks = SparseMatrixAssembler(Xb,Yb)
test_sparse_matrix_assembler(assem_blocks,bmatdata,bvecdata,bdata)

m1 = nz_counter(get_matrix_builder(assem_blocks),(get_rows(assem_blocks),get_cols(assem_blocks)))
symbolic_loop_matrix!(m1,assem_blocks,bmatdata)
m2 = nz_allocation(m1)

m1 = nz_counter(get_matrix_builder(assem_blocks),(get_rows(assem_blocks),get_cols(assem_blocks)))
symbolic_loop_matrix!(m1,a,matdata)
m2 = nz_allocation(m1)

A1_blocks = assemble_matrix(assem_blocks,bmatdata)
b1_blocks = assemble_vector(assem_blocks,bvecdata)
@test A1 ≈ A1_blocks
@test b1 ≈ b1_blocks

y1_blocks = similar(b1_blocks)
mul!(y1_blocks,A1_blocks,b1_blocks)
y1 = similar(b1)
mul!(y1,A1,b1)
@test y1_blocks ≈ y1

A2_blocks, b2_blocks = assemble_matrix_and_vector(assem_blocks,bdata)
@test A2_blocks ≈ A2
@test b2_blocks ≈ b2

A3_blocks = allocate_matrix(assem_blocks,bmatdata)
b3_blocks = allocate_vector(assem_blocks,bvecdata)
assemble_matrix!(A3_blocks,assem_blocks,bmatdata)
assemble_vector!(b3_blocks,assem_blocks,bvecdata)
@test A3_blocks ≈ A1
@test b3_blocks ≈ b1_blocks

A4_blocks, b4_blocks = allocate_matrix_and_vector(assem_blocks,bdata)
assemble_matrix_and_vector!(A4_blocks,b4_blocks,assem_blocks,bdata)
@test A4_blocks ≈ A2_blocks
@test b4_blocks ≈ b2_blocks

############################################################################################

op = AffineFEOperator(biform,liform,X,Y)
block_op = AffineFEOperator(biform,liform,Xb,Yb)

@test get_matrix(op) ≈ get_matrix(block_op)
@test get_vector(op) ≈ get_vector(block_op)

# end # module
