get_extension(f::SingleFieldFESpace) = @notimplemented
get_extension(f::ExtensionFESpace) = f.extension
get_extension(f::UnEvalTrialFESpace{<:ExtensionFESpace}) = get_extension(f.space)
get_extension(f::SingleFieldParamFESpace{<:ExtensionFESpace}) = get_extension(f.space)
get_extension(f::MultiFieldFESpace) = map(get_extension,f.spaces)

FESpaces.num_rows(ext::Extension) = num_free_dofs(get_fe_space(ext))
FESpaces.num_cols(ext::Extension) = num_free_dofs(get_fe_space(ext))

struct ExtensionAssembler <: SparseMatrixAssembler
  assem::SparseMatrixAssembler
  extension::Extension
  trial_fdof_to_bg_fdofs::AbstractVector
  test_fdof_to_bg_fdofs::AbstractVector
end

function ExtensionAssembler(trial::FESpace,test::FESpace)
  bg_trial = get_bg_fe_space(trial)
  bg_test = get_bg_fe_space(test)
  assem = SparseMatrixAssembler(bg_trial,bg_test)
  extension = get_extension(trial) # must be equal to the test extension
  trial_fdof_to_bg_fdofs = get_in_fdof_to_bg_fdofs(trial)
  test_fdof_to_bg_fdofs = get_in_fdof_to_bg_fdofs(test)
  ExtensionAssembler(assem,extension,trial_fdof_to_bg_fdofs,test_fdof_to_bg_fdofs)
end

FESpaces.get_vector_type(a::ExtensionAssembler) = get_vector_type(a.assem)
FESpaces.get_matrix_type(a::ExtensionAssembler) = get_matrix_type(a.assem)
FESpaces.num_rows(a::ExtensionAssembler) = FESpaces.num_rows(a.assem) + FESpaces.num_rows(a.extension)
FESpaces.num_cols(a::ExtensionAssembler) = FESpaces.num_cols(a.assem) + FESpaces.num_cols(a.extension)
FESpaces.get_rows(a::ExtensionAssembler) = Base.OneTo(FESpaces.num_rows(a))
FESpaces.get_cols(a::ExtensionAssembler) = Base.OneTo(FESpaces.num_cols(a))
FESpaces.get_assembly_strategy(a::ExtensionAssembler) = FESpaces.get_assembly_strategy(a.assem)
FESpaces.get_matrix_builder(a::ExtensionAssembler)= get_matrix_builder(a.assem)
FESpaces.get_vector_builder(a::ExtensionAssembler) = get_vector_builder(a.assem)

function extend_incut_vecdata(a::ExtensionAssembler,incut_vecdata)
  cellvals,cellrows = incut_vecdata
  out_cellvals,out_cellrows = a.extension.vecdata
  parent_cellrows = map(rows -> ParentCellDofIds(rows,a.test_fdof_to_bg_fdofs),cellrows)
  parent_out_cellrows = ParentCellDofIds(out_cellrows,a.test_fdof_to_bg_fdofs)
  push!(cellvals,out_cellvals)
  push!(parent_cellrows,parent_out_cellrows)
  return (cellvals,parent_cellrows)
end

function extend_incut_matdata(a::ExtensionAssembler,incut_matdata)
  cellvals,cellrows,cellcols = incut_matdata
  out_cellvals,out_cellrows,out_cellcols = a.extension.matdata
  parent_cellrows = map(rows -> ParentCellDofIds(rows,a.test_fdof_to_bg_fdofs),cellrows)
  parent_out_cellrows = ParentCellDofIds(out_cellrows,a.test_fdof_to_bg_fdofs)
  parent_cellcols = map(cols -> ParentCellDofIds(cols,a.trial_fdof_to_bg_fdofs),cellcols)
  parent_out_cellcols = ParentCellDofIds(out_cellcols,a.trial_fdof_to_bg_fdofs)
  push!(cellvals,out_cellvals)
  push!(parent_cellrows,parent_out_cellrows)
  push!(parent_cellcols,parent_out_cellcols)
  return (cellvals,parent_cellrows,parent_cellcols)
end

function FESpaces.allocate_vector(a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_incut_vecdata(a,vecdata)
  allocate_vector(a.assem,bg_vecdata)
end

function FESpaces.assemble_vector(a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_incut_vecdata(a,vecdata)
  assemble_vector(a.assem,bg_vecdata)
end

function FESpaces.assemble_vector!(b,a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_incut_vecdata(a,vecdata)
  assemble_vector!(b,a.assem,bg_vecdata)
  b
end

function FESpaces.assemble_vector_add!(b,a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_incut_vecdata(a,vecdata)
  assemble_vector_add!(b,a.assem,bg_vecdata)
  b
end

function FESpaces.allocate_matrix(a::ExtensionAssembler,matdata)
  bg_matdata = extend_incut_matdata(a,matdata)
  allocate_matrix(a.assem,bg_matdata)
end

function FESpaces.assemble_matrix(a::ExtensionAssembler,matdata)
  bg_matdata = extend_incut_matdata(a,matdata)
  assemble_matrix(a.assem,bg_matdata)
end

function FESpaces.assemble_matrix!(A,a::ExtensionAssembler,matdata)
  bg_matdata = extend_incut_matdata(a,matdata)
  assemble_matrix!(A,a.assem,bg_matdata)
  A
end

function FESpaces.assemble_matrix_add!(A,a::ExtensionAssembler,matdata)
  bg_matdata = extend_incut_matdata(a,matdata)
  assemble_matrix_add!(A,a.assem,bg_matdata)
  A
end

# utils

struct ParentCellDofIds{A<:AbstractArray,I<:AbstractVector} <: Map
  cell_dof_ids::A
  ids_to_parent_ids::I
end

function Arrays.return_cache(k::ParentCellDofIds,i::Int)
  array_cache(k.cell_dof_ids)
end

function Arrays.evaluate!(c,k::ParentCellDofIds,i::Int)
  ids = getindex!(c,k.cell_dof_ids,i)
  for (j,idsj) in enumerate(ids)
    ids[j] = k.ids_to_parent_ids[ids]
  end
  return ids
end

# function get_in_dof_to_bg_rows(a::ExtensionAssembler)
#   setdiff(FESpaces.get_rows(a),a.extension.dof_to_bg_dofs)
# end

# function get_out_dof_to_bg_rows(a::ExtensionAssembler)
#   get_out_dof_to_bg_dofs(a.extension)
# end

# function get_in_dof_to_bg_cols(a::ExtensionAssembler)
#   setdiff(FESpaces.get_cols(a),a.extension.dof_to_bg_dofs)
# end

# function get_out_dof_to_bg_cols(a::ExtensionAssembler)
#   get_out_dof_to_bg_dofs(a.extension)
# end

# function get_bg_dof_to_in_rows(a::ExtensionAssembler)
#   in_bg_rows = get_in_dof_to_bg_rows(a)
#   bg_in_rows = zeros(eltype(in_bg_rows),FESpaces.num_rows(a))
#   for (row,bg_row) in enumerate(in_bg_rows)
#     bg_in_rows[bg_row] = row
#   end
#   bg_in_rows
# end

# function get_bg_dof_to_out_rows(a::ExtensionAssembler)
#   out_bg_rows = get_out_dof_to_bg_rows(a)
#   bg_out_rows = zeros(eltype(out_bg_rows),FESpaces.num_rows(a))
#   for (row,bg_row) in enumerate(out_bg_rows)
#     bg_out_rows[bg_row] = row
#   end
#   bg_out_rows
# end

# function get_bg_dof_to_in_cols(a::ExtensionAssembler)
#   in_bg_cols = get_in_dof_to_bg_cols(a)
#   bg_in_cols = zeros(eltype(in_bg_cols),FESpaces.num_cols(a))
#   for (col,bg_col) in enumerate(in_bg_cols)
#     bg_in_cols[bg_col] = col
#   end
#   bg_in_cols
# end

# function get_bg_dof_to_out_cols(a::ExtensionAssembler)
#   out_bg_cols = get_out_dof_to_bg_cols(a)
#   bg_out_cols = zeros(eltype(out_bg_cols),FESpaces.num_cols(a))
#   for (col,bg_col) in enumerate(out_bg_cols)
#     bg_out_cols[bg_col] = col
#   end
#   bg_out_cols
# end

# function ordered_insert(
#   A::AbstractVector,
#   B::AbstractVector,
#   row_B_to_row_AB::AbstractVector
#   )

#   AB = similar(A,(length(A)+length(B),))
#   fill!(AB,zero(eltype(AB)))
#   for (row_B,row_AB) in enumerate(row_B_to_row_AB)
#     AB[row_AB] = B[row_B]
#   end
#   AB
# end

# function ordered_insert(
#   A::ConsecutiveParamVector,
#   B::ConsecutiveParamVector,
#   row_B_to_row_AB::AbstractVector
#   )

#   @assert param_length(A) == param_length(B)
#   plength = param_length(A)
#   data_A = get_all_data(A)
#   data_B = get_all_data(B)
#   data_AB = similar(data_A,(size(data_A,1)+size(data_B,1),plength))
#   fill!(data_AB,zero(eltype(data_AB)))
#   for (row_B,row_AB) in enumerate(row_B_to_row_AB)
#     for k in 1:plength
#       data_AB[row_AB,k] = data_B[row_B,k]
#     end
#   end
#   ConsecutiveParamArray(data_AB)
# end

# #TODO fix this!
# function ordered_blockdiag(
#   A::SparseMatrixCSC{Tv,Ti},
#   B::SparseMatrixCSC{Tv,Ti},
#   row_A_to_row_AB::AbstractVector,
#   col_A_to_col_AB::AbstractVector,
#   row_B_to_row_AB::AbstractVector,
#   col_B_to_col_AB::AbstractVector
#   ) where {Tv,Ti}

#   row_AB = vcat(row_A_to_row_AB,row_B_to_row_AB)
#   col_AB = vcat(col_A_to_col_AB,col_B_to_col_AB)
#   AB = blockdiag(A,B)
#   AB[sortperm(row_AB),sortperm(col_AB)]
# end

# #TODO fix this!
# function ordered_blockdiag(
#   A::ConsecutiveParamSparseMatrixCSC,
#   B::ConsecutiveParamSparseMatrixCSC,
#   args...
#   )

#   @assert param_length(A) == param_length(B)
#   ParamArray(map(i -> ordered_blockdiag(param_getindex(A,i),param_getindex(B,i),args...),param_eachindex(A)))
# end

# internal_view(a::AbstractArray,i::AbstractVector...) = InternalView(a,i)

# struct InternalView{T,N,A<:AbstractArray{T,N},I<:AbstractVector} <: AbstractArray{T,N}
#   parent::A
#   i_to_parent_i::NTuple{N,I}
# end

# Base.size(a::InternalView{T,N}) where {T,N} = ntuple(j->length(a.i_to_parent_i[j]),Val{N}())

# function Base.getindex(a::InternalView{T,N},i::Vararg{Int,N}) where {T,N}
#   a.parent[Base.reindex(a.i_to_parent_i,i)...]
# end

# function Base.setindex!(a::InternalView{T,N},v,i::Vararg{Int,N}) where {T,N}
#   a.parent[Base.reindex(a.i_to_parent_i,i)...] = v
# end

# function Algebra.add_entry!(combine::Function,A::InternalView{T,1},v,i) where T
#   parent = A.parent
#   i_to_parent_i, = A.i_to_parent_i
#   parent_i = i_to_parent_i[i]
#   Algebra.add_entry!(combine,parent,v,parent_i)
# end

# function Algebra.add_entry!(combine::Function,A::InternalView{T,2},v,i,j) where T
#   parent = A.parent
#   i_to_parent_i,j_to_parent_j = A.i_to_parent_i
#   parent_i = i_to_parent_i[i]
#   parent_j = j_to_parent_j[j]
#   Algebra.add_entry!(combine,parent,v,parent_i,parent_j)
# end

# Base.fill!(a::InternalView,v) = LinearAlgebra.fill!(a.parent,v)
# LinearAlgebra.fillstored!(a::InternalView,v) = LinearAlgebra.fillstored!(a.parent,v)

# const ParamInternalView{T,N,I<:AbstractVector} = InternalView{T,N,<:AbstractParamArray,I}

# Base.size(a::ParamInternalView{T,N}) where {T,N} = size(a.parent)

# function Base.getindex(a::ParamInternalView{T,N},i::Vararg{Int,N}) where {T,N}
#   getindex(a.parent,i...)
# end

# function Base.setindex!(a::ParamInternalView{T,N},v,i::Vararg{Int,N}) where {T,N}
#   setindex!(a.parent,v,i...)
# end
