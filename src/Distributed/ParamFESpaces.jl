const DistributedSingleFieldParamFESpace = DistributedSingleFieldFESpace{<:AbstractVector{<:SingleFieldParamFESpace{S}},B,C} where {S,B,C}

function GridapDistributed._find_vector_type(
  spaces::AbstractVector{<:SingleFieldParamFESpace},gids)
  T = get_vector_type(PartitionedArrays.getany(spaces))
  if isa(gids,PRange)
    vector_type = typeof(PVector{T}(undef,partition(gids)))
  else
    vector_type = typeof(BlockPVector{T}(undef,gids))
  end
  return vector_type
end

function FEM.TrialParamFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.TrialParamFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s,fun)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.TrialParamFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(fun,s)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.TrialParamFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace!(s,fun)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.HomogeneousTrialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    HomogeneousTrialParamFESpace(s,args...)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.TrivialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    TrivialParamFESpace(s,args...)
  end
  vector_types = GridapDistributed._find_vector_type(spaces,f.gids)
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.get_polynomial_order(f::DistributedFESpace)
  FEM.get_polynomial_order(PartitionedArrays.getany(local_views(f)))
end

function _to_distributed_fe_space(trial::TransientTrialParamFESpace{<:DistributedSingleFieldFESpace})
  map(local_views(trial.space)) do space
    TransientTrialParamFESpace(space,trial.dirichlet)
  end
end

function FEM.collect_cell_matrix_for_trian(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::DistributedDomainContribution,
  trian::DistributedTriangulation)
  map(
    FEM.collect_cell_matrix_for_trian,
    local_views(trial),
    local_views(test),
    local_views(a),
    local_views(trian))
end

function FEM.collect_cell_vector_for_trian(
  test::DistributedFESpace,
  a::DistributedDomainContribution,
  trian::DistributedTriangulation)
  map(
    FEM.collect_cell_vector_for_trian,
    local_views(test),
    local_views(a),
    local_views(trian))
end

const DistributedSingleFieldParamFEFunction = GridapDistributed.DistributedCellField{A,T} where {A<:AbstractVector{<:SingleFieldParamFEFunction},T}

function FESpaces.SparseMatrixAssembler(
  trial::DistributedSingleFieldParamFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows())

  Tpv = PartitionedArrays.getany(map(get_vector_type,local_views(trial)))
  T  = eltype(Tpv)
  Tm = SparseMatrixCSC{T,Int}
  Tpm = typeof(ParamMatrix{Tm}(undef,param_length(trial)))
  SparseMatrixAssembler(Tpm,Tpv,trial,test,par_strategy)
end

function FEM.get_param_vector_builder(
  a::DistributedSparseMatrixAssembler,
  r::FEM.AbstractParamRealization)

  L = length(r)
  vec = get_vector_builder(a)
  V = get_array_type(vec)
  elV = eltype(V)
  pvector_type = ParamVector{elV,Vector{V},L}
  ArrayBuilder(pvector_type)
end

function FEM.get_param_assembler(
  a::DistributedSparseMatrixAssembler,
  r::FEM.AbstractParamRealization)

  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  assems = map(local_views(a)) do assem
    FEM.get_param_assembler(assem,r)
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
