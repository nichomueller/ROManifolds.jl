function FEM.TrialParamFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(TrialParamFESpace,f.spaces)
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialParamFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialParamFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(fun,s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialParamFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace!(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.HomogeneousTrialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    HomogeneousTrialParamFESpace(s,args...)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FESpaces.zero_free_values(
  f::DistributedSingleFieldFESpace{<:AbstractArray{<:TrialParamFESpace}})

  index_partition = partition(f.gids)
  vector_partition = map(local_views(f)) do fi
    Gridap.FESpaces.zero_free_values(fi)
  end
  PVector(vector_partition,index_partition)
end

function FESpaces.collect_cell_matrix(
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

function FESpaces.collect_cell_vector(
  test::DistributedFESpace,
  a::DistributedDomainContribution,
  trian::DistributedTriangulation)
  map(
    collect_cell_vector,
    local_views(test),
    local_views(a),
    local_views(trian))
end
