abstract type RBAssemblyStrategy <: AssemblyStrategy end
struct SpaceOnly <: RBAssemblyStrategy end
struct SpaceTime <: RBAssemblyStrategy end

struct RBAssembler{M,V,S} <: Assembler
  matrix_builder::M # algebraic contributions
  vector_builder::V # algebraic contributions
  strategy::RBAssemblyStrategy       # space or space-time
end

FESpaces.get_matrix_builder(a::RBAssembler) = a.matrix_builder

FESpaces.get_vector_builder(a::RBAssembler) = a.vector_builder

FESpaces.get_assembly_strategy(a::RBAssembler) = a.strategy

function FESpaces.allocate_vector(a::RBAssembler,vecdata)
  allocate_vector(eltype(get_vector_builder(a)),get_rows(a))
end

function FESpaces.assemble_vector!(b,a::RBAssembler,vecdata)
  fill!(b,zero(eltype(b)))
  assemble_vector_add!(b,a,vecdata)
end

function FESpaces.assemble_vector_add!(b,a::RBAssembler,vecdata)
  numeric_loop_vector!(b,get_vector_builder(a),vecdata)
  create_from_nz(b)
end

function FESpaces.allocate_matrix(a::RBAssembler,matdata)
  allocate_matrix(eltype(get_matrix_builder(a)),get_rows(a),get_cols(a))
end

function FESpaces.assemble_matrix!(mat,a::RBAssembler,matdata)
  LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
  assemble_matrix_add!(mat,get_matrix_builder(a),matdata)
end

function FESpaces.assemble_matrix_add!(mat,a::RBAssembler,matdata)
  numeric_loop_matrix!(mat,a,matdata)
  create_from_nz(mat)
end

function FESpaces.numeric_loop_vector!(b,a::RBAssembler,vecdata)
  bcoeff,bcontrib = b
  strategy = get_assembly_strategy(a)
  builder = get_vector_builder(a)
  @check length(builder) == length(vecdata)
  v = map(eachindex(builder)) do i
    coeff = rb_coefficient!(bcoeff,strategy,builder[i],vecdata[i])
    rb_contribution!(bcontrib,builder[i],coeff)
  end
  sum(v)
end

function FESpaces.numeric_loop_matrix!(A,a::RBAssembler,matdata)
  Acoeff,Acontrib = A
  strategy = get_assembly_strategy(a)
  builder = get_matrix_builder(a)
  @check length(builder) == length(matdata)
  v = map(eachindex(builder)) do i
    coeff = rb_coefficient!(Acoeff,strategy,builder[i],matdata[i])
    rb_contribution!(Acontrib,builder[i],coeff)
  end
  sum(v)
end
