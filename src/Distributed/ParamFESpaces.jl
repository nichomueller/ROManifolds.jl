const DistributedSingleFieldParamFESpace = DistributedSingleFieldFESpace{<:AbstractVector{<:SingleFieldParamFESpace{S}},B,C} where {S,B,C}

function GridapDistributed._find_vector_type(
  spaces::AbstractVector{<:SingleFieldParamFESpace},gids;split_own_and_ghost=false)
  T = get_vector_type(PartitionedArrays.getany(spaces))
  if split_own_and_ghost
    T = OwnAndGhostVectors{T}
  end
  if isa(gids,PRange)
    vector_type = typeof(PVector{T}(undef,partition(gids)))
  else
    vector_type = typeof(BlockPVector{T}(undef,gids))
  end
  return vector_type
end

function ParamFESpaces.TrialParamFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function ParamFESpaces.TrialParamFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s,fun)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function ParamFESpaces.TrialParamFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(fun,s)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function ParamFESpaces.TrialParamFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace!(s,fun)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function ParamFESpaces.HomogeneousTrialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    HomogeneousTrialParamFESpace(s,args...)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function ParamFESpaces.TrivialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    TrivialParamFESpace(s,args...)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function ParamFESpaces.collect_cell_matrix_for_trian(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::DistributedDomainContribution,
  trian::DistributedTriangulation)

  map(collect_cell_matrix_for_trian,
    local_views(trial),
    local_views(test),
    local_views(a),
    local_views(trian))
end

function ParamFESpaces.collect_cell_vector_for_trian(
  test::DistributedFESpace,
  a::DistributedDomainContribution,
  trian::DistributedTriangulation)

  map(collect_cell_vector_for_trian,
    local_views(test),
    local_views(a),
    local_views(trian))
end

# const DistributedSingleFieldParamFEFunction{A,B} = DistributedSingleFieldFEFunction{A,B,<:AbstractVector{<:AbstractParamVector}}

function ParamFESpaces.get_param_matrix_builder(
  a::DistributedSparseMatrixAssembler,
  r::AbstractRealization)

  mat = get_matrix_builder(a)
  M = get_array_type(mat)
  T = eltype(M)
  L = length(r)
  pmatrix_type = MatrixOfSparseMatricesCSC{T,Int,L}
  SparseMatrixBuilder(pmatrix_type)
end

function ParamFESpaces.get_param_vector_builder(
  a::DistributedSparseMatrixAssembler,
  r::AbstractRealization)

  vec = get_vector_builder(a)
  V = get_array_type(vec)
  T = eltype(V)
  L = length(r)
  pvector_type = ConsecutiveVectorOfVectors{T,L}
  ArrayBuilder(pvector_type)
end

function ParamFESpaces.get_param_assembler(
  a::DistributedSparseMatrixAssembler,
  r::FEM.AbstractRealization)

  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  assems = map(local_views(a)) do assem
    get_param_assembler(assem,r)
  end
  assem = PartitionedArrays.getany(assems)
  local_mat_type = FESpaces.get_matrix_type(assem)
  local_vec_type = FESpaces.get_vector_type(assem)
  par_strategy = FESpaces.get_assembly_strategy(a)
  matrix_builder = PSparseMatrixBuilderCOO(local_mat_type,par_strategy)
  vector_builder = PVectorBuilder(local_vec_type,par_strategy)
  DistributedSparseMatrixAssembler(par_strategy,assems,matrix_builder,vector_builder,rows,cols)
end

const DistributedMultiFieldParamFESpace = GridapDistributed.DistributedMultiFieldFESpace{MS,<:AbstractVector{<:MultiFieldParamFESpace},B,C,D} where {MS,B,C,D}
const DistributedMultiFieldParamFEFunction = GridapDistributed.DistributedMultiFieldFEFunction{<:AbstractVector{<:SingleFieldParamFEFunction},B,C} where {B,C}
const DistributedParamFESpace = Union{DistributedSingleFieldParamFESpace,DistributedMultiFieldParamFESpace}
