# module BlockMatrixAssemblersTests
using Test, BlockArrays, SparseArrays, LinearAlgebra

using Gridap
using Gridap.Algebra, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using Mabla.FEM

############################################################################################
parametric = true
μ = PRealization([[1],[2],[3]])
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
  U = TrialPFESpace(V,solμ)
  biform((u1,u2),(v1,v2)) = ∫(solμ*∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
  liform((v1,v2)) = ∫(solμ*v1 - v2)*dΩ
else
  U = TrialFESpace(V,f)
  biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
  liform((v1,v2)) = ∫(v1 - v2)*dΩ
end

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
FEM.test_passembler(assem_blocks,bmatdata,bvecdata,bdata)

A1_blocks = assemble_matrix(assem_blocks,bmatdata)
b1_blocks = assemble_vector(assem_blocks,bvecdata)
for i = 1:length(solμ)
  @test A1[i] ≈ A1_blocks[i]#mortar(getindex.(A1_blocks.blocks,i))
  @test b1[i] ≈ b1_blocks[i]#mortar(getindex.(b1_blocks.blocks,i))
end

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

############
#typeof(A1_blocks) <: BlockMatrix{Float64,<:AbstractMatrix{<:ParamMatrix{Float64, 3}}}

# const ParamBlockMatrix{T,L} = BlockMatrix{T,<:AbstractMatrix{<:ParamMatrix{Float64,L}}}
# const ParamBlockVector{T,L} = BlockVector{T,<:AbstractVector{<:ParamVector{Float64,L}}}

# Base.getindex(a::ParamBlockMatrix{T,N},i::Vararg{Integer,N}) = hvcat(map(a->getindex(blocks(a),i),a)...)
# Base.getindex(a::ParamBlockVector,i...) = vcat(getindex.(blocks(a),i...)...)

# @assert typeof(A1_blocks) <: ParamBlockMatrix{Float64,3}
# @assert typeof(b1_blocks) <: ParamBlockVector{Float64,3}

# _dio(a,i) = map(a->getindex(blocks(a),i),a)
# _dio(A1_blocks,1)

function Base.length(
  a::BlockArray{T,N,<:AbstractArray{ParamArray{T,N,A,L}}}
  ) where {T,N,A,L,BS}
  length(first(blocks(a)))
end

function Base.getindex(
  a::BlockArray{T,N,<:AbstractArray{ParamArray{T,N,A,L}},BS},
  i::Integer) where {T,N,A,L,BS}
  # blocksi = map(blocks(a)) do block
  #   block[i]
  # end
  blocksi = getindex.(blocks(a),i)
  mortar(blocksi)
end

A1_blocks[1]

m = sparse(rand(3,3))
mm = Matrix{typeof(m)}(undef,2,2)
mm[1] = m; mm[2] = m; mm[3] = m; mm[4] = m;
A = ParamArray(mm)
AA = mortar(mm)

v = rand(3)
vv = [v,v]
bb = Vector{typeof(vv)}(undef,2)
bb[1] = vv; bb[2] = vv
b = mortar(bb)

A*b
