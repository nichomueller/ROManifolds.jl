function FEM.TrialPFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(TrialPFESpace,f.spaces)
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialPFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialPFESpace(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialPFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialPFESpace(fun,s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialPFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialPFESpace!(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.HomogeneousTrialPFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    HomogeneousTrialPFESpace(s,args...)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FESpaces.zero_free_values(
  f::DistributedSingleFieldFESpace{<:AbstractArray{<:TrialPFESpace}})

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
