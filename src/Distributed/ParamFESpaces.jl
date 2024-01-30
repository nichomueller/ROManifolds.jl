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
  spaces,vector_types = map(f.spaces) do s
    space = TrialParamFESpace(s)
    vector_type = get_vector_type(space)
    space,vector_type
  end |> tuple_of_arrays
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
  spaces,vector_types = map(f.spaces) do s
    space = TrialParamFESpace!(s,fun)
    vector_type = get_vector_type(space)
    space,vector_type
  end |> tuple_of_arrays
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.HomogeneousTrialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces,vector_types = map(f.spaces) do s
    space = HomogeneousTrialParamFESpace(s,args...)
    vector_type = get_vector_type(space)
    space,vector_type
  end |> tuple_of_arrays
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.FESpaceToParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces,vector_types = map(f.spaces) do s
    space = FESpaceToParamFESpace(s,args...)
    vector_type = get_vector_type(space)
    space,vector_type
  end |> tuple_of_arrays
  DistributedSingleFieldFESpace(spaces,f.gids,vector_types)
end

function FEM.length_dirichlet_values(f::DistributedSingleFieldFESpace)
  length_dirichlet_values(PartitionedArrays.getany(local_views(f)))
end

function FEM.collect_cell_matrix_for_trian(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::DistributedDomainContribution,
  trian::DistributedTriangulation)
  map(
    collect_cell_matrix,
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
    collect_cell_vector,
    local_views(test),
    local_views(a),
    local_views(trian))
end

const DistributedSingleFieldParamFEFunction = GridapDistributed.DistributedCellField{A,T} where {A<:AbstractVector{<:SingleFieldParamFEFunction},T}

function FESpaces.SparseMatrixAssembler(
  trial::DistributedSingleFieldParamFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows())

  Tv = PartitionedArrays.getany(map(get_vector_type,local_views(trial)))
  T  = eltype(Tv)
  Tm = SparseMatrixCSC{T,Int}
  L = length_free_values(trial)
  Tpm = typeof(ParamMatrix{Tm}(undef,L))
  Tpv = typeof(ParamVector{Tv}(undef,L))
  SparseMatrixAssembler(Tpm,Tpv,trial,test,par_strategy)
end
