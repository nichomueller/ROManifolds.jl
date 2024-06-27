function TProductBlockSparseMatrixAssembler(
  mat,
  vec,
  trial::MultiFieldFESpace{MS},
  test::MultiFieldFESpace{MS},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where MS <: BlockMultiFieldStyle

  assem = SparseMatrixAssembler(mat,vec,trial.space,test.space,strategy)
  assems_1d = map((U,V)->SparseMatrixAssembler(mat,vec,U,V,strategy),trial.spaces_1d,test.spaces_1d)
  row_index_map = get_tp_dof_index_map(test)
  col_index_map = get_tp_dof_index_map(trial)
  TProductSparseMatrixAssembler(assem,assems_1d,row_index_map,col_index_map)
end

function TProductBlockSparseMatrixAssembler(
  trial::MultiFieldFESpace{MS},
  test::MultiFieldFESpace{MS}
  ) where MS <: BlockMultiFieldStyle

  T = get_dof_value_type(trial)
  matrix_type = SparseMatrixCSC{T,Int}
  vector_type = Vector{T}
  TProductBlockSparseMatrixAssembler(matrix_type,vector_type,trial,test)
end

function get_tp_triangulation(f::MultiFieldFESpace)
  get_tp_triangulation(f.spaces[1])
end

function get_tp_fe_basis(f::MultiFieldFESpace)
  nfields = length(f.spaces)
  all_febases = MultiFieldTProductFEBasisComponent[]
  for field_i in 1:nfields
    dv_i = get_tp_fe_basis(f.spaces[field_i])
    @assert BasisStyle(dv_i) == TestBasis()
    dv_i_b = MultiFieldTProductFEBasisComponent(dv_i,field_i,nfields)
    push!(all_febases,dv_i_b)
  end
  MultiFieldCellField(all_febases)
end

function get_tp_trial_fe_basis(f::MultiFieldFESpace)
  nfields = length(f.spaces)
  all_febases = MultiFieldTProductFEBasisComponent[]
  for field_i in 1:nfields
    du_i = get_tp_trial_fe_basis(f.spaces[field_i])
    @assert BasisStyle(du_i) == TrialBasis()
    du_i_b = MultiFieldTProductFEBasisComponent(du_i,field_i,nfields)
    push!(all_febases,du_i_b)
  end
  MultiFieldCellField(all_febases)
end
