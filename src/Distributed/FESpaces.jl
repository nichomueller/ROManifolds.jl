function FEM.TrialPFESpace(f::DistributedSingleFieldFESpace{<:SingleFieldPFESpace})
  spaces = map(TrialPFESpace,f.spaces)
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialPFESpace(f::DistributedSingleFieldFESpace{<:SingleFieldPFESpace},fun)
  spaces = map(f.spaces) do s
    TrialPFESpace(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialPFESpace(fun,f::DistributedSingleFieldFESpace{<:SingleFieldPFESpace})
  spaces = map(f.spaces) do s
    TrialPFESpace(fun,s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FEM.TrialPFESpace!(f::DistributedSingleFieldFESpace{<:SingleFieldPFESpace},fun)
  spaces = map(f.spaces) do s
    TrialPFESpace!(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
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

function FEM.PTSparseMatrixAssembler(assem::DistributedSparseMatrixAssembler,μ,t)
  len = FEM._length(μ,t)
  DistributedSparseMatrixAssembler(
    assem.par_strategy,
    assem.assems,
    SparsePTMatrixBuilder(assem.matrix_builder,len),
    PTArrayBuilder(assem.vector_builder,len),
    assem.test_dofs_gids_prange,
    assem.trial_dofs_gids_prange)
end
