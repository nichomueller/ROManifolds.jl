const ExtensionFESpace = Union{SingleFieldExtensionFESpace,MultiFieldExtensionFESpace}

get_extension(f::ExtensionFESpace) = @abstractmethod
get_internal_space(f::ExtensionFESpace) = @abstractmethod
get_external_space(f::ExtensionFESpace) = @abstractmethod

struct ExtensionAssembler <: SparseMatrixAssembler
  int_assem::SparseMatrixAssembler
  extension::Extension
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::ExtensionFESpace,
  test::ExtensionFESpace,
  strategy::AssemblyStrategy=FESpaces.DefaultAssemblyStrategy()
  )

  int_trial = get_internal_space(trial)
  int_test = get_internal_space(test)
  int_assem = SparseMatrixAssembler(mat,vec,int_trial,int_test,strategy)
  extension = get_extension(trial)
  ExtensionAssembler(int_assem,extension)
end

FESpaces.get_vector_type(a::ExtensionAssembler) = get_vector_type(a.int_assem)
FESpaces.get_matrix_type(a::ExtensionAssembler) = get_matrix_type(a.int_assem)
FESpaces.num_rows(a::ExtensionAssembler) = FESpaces.num_rows(a.int_assem) + FESpaces.num_rows(a.extension)
FESpaces.num_cols(a::ExtensionAssembler) = FESpaces.num_cols(a.int_assem) + FESpaces.num_cols(a.extension)

function FESpaces.get_rows(a::ExtensionAssembler)
  @notimplemented
end

function FESpaces.get_cols(a::ExtensionAssembler)
  @notimplemented
end

function FESpaces.get_assembly_strategy(a::ExtensionAssembler)
  get_assembly_strategy(a.int_assem)
end

function FESpaces.get_matrix_builder(a::ExtensionAssembler)
  get_matrix_builder(a.int_assem)
end

function FESpaces.get_vector_builder(a::ExtensionAssembler)
  get_vector_builder(a.int_assem)
end

function FESpaces.allocate_vector(a::ExtensionAssembler,vecdata)
  int_v = allocate_vector(a.int_assem,vecdata)
  ext_v = get_extension_vector(a.extension)
  mortar([int_v,ext_v])
end

function FESpaces.assemble_vector!(b,a::ExtensionAssembler,vecdata)
  assemble_vector!(b[Block(1)],a.int_assem,vecdata)
  copyto!(b[Block(2)],get_extension_vector(a.extension))
end

function FESpaces.assemble_vector_add!(b,a::ExtensionAssembler,vecdata)
  assemble_vector_add!(b[Block(1)],a.int_assem,vecdata)
  copyto!(b[Block(2)],get_extension_vector(a.extension))
end

function FESpaces.assemble_vector(a::ExtensionAssembler,vecdata)
  int_v = assemble_vector(a.int_assem,vecdata)
  ext_v = get_extension_vector(a.extension)
  mortar([int_v,ext_v])
end

function FESpaces.allocate_matrix(a::ExtensionAssembler,matdata)
  m = Matrix{get_matrix_type(a)}(undef,2,2)
  m[1,1] = allocate_matrix(a.int_assem,matdata)
  m[2,1] = spzeros(FESpaces.num_cols(a.int_assem),FESpaces.num_rows(a.extension))
  m[1,2] = spzeros(FESpaces.num_rows(a.int_assem),FESpaces.num_cols(a.extension))
  m[2,2] = get_extension_matrix(a.extension)
  mortar(m)
end

function FESpaces.assemble_matrix!(mat,a::ExtensionAssembler,matdata)
  assemble_matrix!(mat[Block(1,1)],a.int_assem,matdata)
  copyto!(mat[Block(2,2)],get_extension_matrix(a.extension))
end

function FESpaces.assemble_matrix_add!(mat,a::ExtensionAssembler,matdata)
  assemble_matrix_add!(mat[Block(1,1)],a.int_assem,matdata)
  copyto!(mat[Block(2,2)],get_extension_matrix(a.extension))
end

function FESpaces.assemble_matrix(a::ExtensionAssembler,matdata)
  m = Matrix{get_matrix_type(a)}(undef,2,2)
  m[1,1] = assemble_matrix(a.int_assem,matdata)
  m[2,1] = spzeros(FESpaces.num_cols(a.int_assem),FESpaces.num_rows(a.extension))
  m[1,2] = spzeros(FESpaces.num_rows(a.int_assem),FESpaces.num_cols(a.extension))
  m[2,2] = get_extension_matrix(a.extension)
  mortar(m)
end

function FESpaces.allocate_matrix_and_vector(a::ExtensionAssembler,data)
  int_A,int_b = allocate_matrix_and_vector(a.int_assem,data)

  A = Matrix{typeof(int_A)}(undef,2,2)
  A[1,1] = int_A
  A[2,1] = spzeros(FESpaces.num_cols(a.int_assem),FESpaces.num_rows(a.extension))
  A[1,2] = spzeros(FESpaces.num_rows(a.int_assem),FESpaces.num_cols(a.extension))
  A[2,2] = get_extension_matrix(a.extension)

  b = Matrix{typeof(int_b)}(undef,2)
  b[1] = ext_b
  b[2] = get_extension_vector(a.extension)

  mortar(A),mortar(b)
end

function FESpaces.assemble_matrix_and_vector!(A,b,a::ExtensionAssembler, data)
  assemble_matrix_and_vector!(A[Block(1,1)],b[Block(1)],a.int_assem,data)
  copyto!(A[Block(2,2)],get_extension_matrix(a.extension))
  copyto!(b[Block(2)],get_extension_vector(a.extension))
end

function FESpaces.assemble_matrix_and_vector_add!(A,b,a::ExtensionAssembler,data)
  assemble_matrix_and_vector_add!(A[Block(1,1)],b[Block(1)],a.int_assem,data)
  copyto!(A[Block(2,2)],get_extension_matrix(a.extension))
  copyto!(b[Block(2)],get_extension_vector(a.extension))
end

function FESpaces.assemble_matrix_and_vector(a::ExtensionAssembler, data)
  int_A,int_b = assemble_matrix_and_vector(a.int_assem,data)

  A = Matrix{typeof(int_A)}(undef,2,2)
  A[1,1] = int_A
  A[2,1] = spzeros(FESpaces.num_rows(a.extension),FESpaces.num_cols(a.int_assem))
  A[1,2] = spzeros(FESpaces.num_rows(a.int_assem),FESpaces.num_cols(a.extension))
  A[2,2] = get_extension_matrix(a.extension)

  b = Vector{typeof(int_b)}(undef,2)
  b[1] = int_b
  b[2] = get_extension_vector(a.extension)

  mortar(A),mortar(b)
end
