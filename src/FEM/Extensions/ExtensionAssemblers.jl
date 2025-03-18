get_extension(f::SingleFieldFESpace) = @notimplemented
get_extension(f::ExtensionFESpace) = f.extension
get_extension(f::UnEvalTrialFESpace{<:ExtensionFESpace}) = get_extension(f.space)
get_extension(f::SingelFieldParamFESpace{<:ExtensionFESpace}) = get_extension(f.space)
get_extension(f::MultiFieldFESpace) = map(get_extension,f.spaces)

FESpaces.num_rows(ext::Extension) = num_free_dofs(get_fe_space(ext))
FESpaces.num_cols(ext::Extension) = num_free_dofs(get_fe_space(ext))

struct ExtensionAssembler <: SparseMatrixAssembler
  assem::SparseMatrixAssembler
  extension::Extension
end

function ExtensionAssembler(trial::FESpace,test::FESpace)
  assem = SparseMatrixAssembler(trial,test)
  extension = get_extension(trial)
  ExtensionAssembler(assem,extension)
end

FESpaces.get_vector_type(a::ExtensionAssembler) = get_vector_type(a.assem)
FESpaces.get_matrix_type(a::ExtensionAssembler) = get_matrix_type(a.assem)
FESpaces.num_rows(a::ExtensionAssembler) = FESpaces.num_rows(a.assem) + FESpaces.num_rows(a.extension)
FESpaces.num_cols(a::ExtensionAssembler) = FESpaces.num_cols(a.assem) + FESpaces.num_cols(a.extension)
FESpaces.get_rows(a::ExtensionAssembler) = Base.OneTo(FESpaces.num_rows(a))
FESpaces.get_cols(a::ExtensionAssembler) = Base.OneTo(FESpaces.num_cols(a))
FESpaces.get_assembly_strategy(a::ExtensionAssembler) = get_assembly_strategy(a.assem)
FESpaces.get_matrix_builder(a::ExtensionAssembler)= get_matrix_builder(a.assem)
FESpaces.get_vector_builder(a::ExtensionAssembler) = get_vector_builder(a.assem)

function get_in_dof_to_bg_rows(a::ExtensionAssembler)
  setdiff(FESpaces.get_rows(a),a.extension.dof_to_bg_dofs)
end

function get_out_dof_to_bg_rows(a::ExtensionAssembler)
  get_out_dof_to_bg_dofs(a.extension)
end

function get_in_dof_to_bg_cols(a::ExtensionAssembler)
  setdiff(FESpaces.get_cols(a),a.extension.dof_to_bg_dofs)
end

function get_out_dof_to_bg_cols(a::ExtensionAssembler)
  get_out_dof_to_bg_dofs(a.extension)
end

function get_bg_dof_to_in_rows(a::ExtensionAssembler)
  in_bg_rows = get_in_dof_to_bg_rows(a)
  bg_in_rows = zeros(eltype(in_bg_rows),FESpaces.num_rows(a))
  for (row,bg_row) in enumerate(in_bg_rows)
    bg_in_rows[bg_row] = row
  end
  bg_in_rows
end

function get_bg_dof_to_out_rows(a::ExtensionAssembler)
  out_bg_rows = get_out_dof_to_bg_rows(a)
  bg_out_rows = zeros(eltype(out_bg_rows),FESpaces.num_rows(a))
  for (row,bg_row) in enumerate(out_bg_rows)
    bg_out_rows[bg_row] = row
  end
  bg_out_rows
end

function get_bg_dof_to_in_cols(a::ExtensionAssembler)
  in_bg_cols = get_in_dof_to_bg_cols(a)
  bg_in_cols = zeros(eltype(in_bg_cols),FESpaces.num_cols(a))
  for (col,bg_col) in enumerate(in_bg_cols)
    bg_in_cols[bg_col] = col
  end
  bg_in_cols
end

function get_bg_dof_to_out_cols(a::ExtensionAssembler)
  out_bg_cols = get_out_dof_to_bg_cols(a)
  bg_out_cols = zeros(eltype(out_bg_cols),FESpaces.num_cols(a))
  for (col,bg_col) in enumerate(out_bg_cols)
    bg_out_cols[bg_col] = col
  end
  bg_out_cols
end

function FESpaces.allocate_vector(a::ExtensionAssembler,vecdata)
  out_bg_rows = get_out_dof_to_bg_rows(a)
  b = zeros(get_vector_type(a),num_rows(a))
  out_b = a.extension.vector
  for (dof,bg_dof) in enumerate(out_bg_rows)
    b[bg_dof] = out_b[dof]
  end
  b
end

function FESpaces.assemble_vector!(b,a::ExtensionAssembler,vecdata)
  in_bg_rows = get_in_dof_to_bg_rows(a)
  in_b = internal_view(b,in_bg_rows)
  assemble_vector!(in_b,a.assem,vecdata)
end

function FESpaces.assemble_vector_add!(b,a::ExtensionAssembler,vecdata)
  in_bg_rows = get_in_dof_to_bg_rows(a)
  in_b = internal_view(b,in_bg_rows)
  assemble_vector_add!(in_b,a.assem,vecdata)
end

function FESpaces.allocate_matrix(a::ExtensionAssembler,matdata)
  out_bg_rows = get_out_dof_to_bg_rows(a)
  out_bg_cols = get_out_dof_to_bg_cols(a)
  in_A = allocate_matrix(a.assem,matdata)
  out_A = a.extension.matrix
  ordered_blockdiag(in_A,out_A,a)
end

function FESpaces.assemble_matrix!(A,a::ExtensionAssembler,matdata)
  in_bg_rows = get_in_dof_to_bg_rows(a)
  in_bg_cols = get_in_dof_to_bg_cols(a)
  in_A = internal_view(A,in_bg_rows,in_bg_cols)
  assemble_matrix!(in_A,a.assem,matdata)
end

function FESpaces.assemble_matrix_add!(A,a::ExtensionAssembler,matdata)
  in_bg_rows = get_in_dof_to_bg_rows(a)
  in_bg_cols = get_in_dof_to_bg_cols(a)
  in_A = internal_view(A,in_bg_rows,in_bg_cols)
  assemble_matrix_add!(in_A,a.assem,matdata)
end

# utils

internal_view(a::AbstractArray,i::AbstractVector...) = InternalView(a,i)

struct InternalView{T,N,A<:AbstractArray{T,N},I<:AbstractVector} <: AbstractArray{T,N}
  parent::A
  i_to_parent_i::NTuple{N,I}
end

Base.size(a::InternalView{T,N}) where {T,N} = ntuple(j->length(a.i_to_parent_i[j]),Val{N}())

function Base.getindex(a::InternalView{T,N},i::Vararg{Int,N}) where {T,N}
  a.parent[Base.reindex(a.i_to_parent_i,i)...]
end

function Base.setindex!(a::InternalView{T,N},v,i::Vararg{Int,N}) where {T,N}
  a.parent[Base.reindex(a.i_to_parent_i,i)...] = v
end

const InternalVectorView{T,I} = InternalView{T,1,Vector{T},I}
const InternalSparseMatrixCSCView{T,I} = InternalView{T,2,SparseMatrixCSC{T,Int},I}

function Algebra.add_entry!(combine::Function,A::InternalVectorView,v,i)
  parent = A.parent
  i_to_parent_i, = A.i_to_parent_i
  parent_i = i_to_parent_i[i]
  Algebra.add_entry!(combine,parent,v,parent_i)
end

function Algebra.add_entry!(combine::Function,A::InternalSparseMatrixCSCView,v,i,j)
  parent = A.parent
  i_to_parent_i,j_to_parent_j = A.i_to_parent_i
  parent_i = i_to_parent_i[i]
  parent_j = j_to_parent_j[j]
  Algebra.add_entry!(combine,parent,v,parent_i,parent_j)
end

const ParamInternalView{T,N,A<:AbstractParamArray{T,N},I<:AbstractVector} = InternalView{T,N,A,I}

Base.size(a::ParamInternalView{T,N}) where {T,N} = size(a.parent)

function Base.getindex(a::ParamInternalView{T,N},i::Vararg{Int,N}) where {T,N}
  getindex(a.parent,i...)
end

function Base.setindex!(a::ParamInternalView{T,N},v,i::Vararg{Int,N}) where {T,N}
  setindex!(a.parent,v,i...)
end

const ParamInternalVectorView{T,I} = InternalView{T,1,<:ConsecutiveParamVector{T},I}
const ParamInternalSparseMatrixCSCView{T,I} = InternalView{T,2,<:ConsecutiveParamSparseMatrix{T},I}

function Algebra.add_entry!(combine::Function,A::ParamInternalVectorView,v,i)
  parent = A.parent
  i_to_parent_i, = A.i_to_parent_i
  parent_i = i_to_parent_i[i]
  Algebra.add_entry!(combine,parent,v,parent_i)
end

function Algebra.add_entry!(combine::Function,A::ParamInternalSparseMatrixCSCView,v,i)
  parent = A.parent
  i_to_parent_i,j_to_parent_j = A.i_to_parent_i
  parent_i = i_to_parent_i[i]
  parent_j = j_to_parent_j[j]
  Algebra.add_entry!(combine,parent,v,parent_i,parent_j)
end
