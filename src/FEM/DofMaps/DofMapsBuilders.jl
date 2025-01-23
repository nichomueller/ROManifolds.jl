function get_dof_map(f::SingleFieldFESpace)
  n = num_free_dofs(f)
  VectorDofMap(n)
end

function get_dof_map(f::MultiFieldFESpace)
  map(get_dof_map,f.spaces)
end

# sparse interface

"""
    get_sparse_dof_map(trial::FESpace,test::FESpace,args...) -> AbstractDofMap

Returns the index maps related to jacobians in a FE problem. The default output
is a `TrivialSparseMatrixDofMap`; when the trial and test spaces are of type
`TProductFESpace`, a `SparseMatrixDofMap` is returned.
"""
function get_sparse_dof_map(trial::SingleFieldFESpace,test::SingleFieldFESpace,args...)
  sparsity = get_sparsity(trial,test,args...)
  get_sparse_dof_map(sparsity,trial,test,args...)
end

function get_sparse_dof_map(trial::MultiFieldFESpace,test::MultiFieldFESpace,args...)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    get_sparse_dof_map(trial[j],test[i],args...)
  end
end

# utils

function get_dof_to_mask(f::FESpace)
  @abstractmethod
end

function get_dof_to_mask(f::SingleFieldFESpace)
  fill(false,num_free_dofs(f))
end

function get_dof_to_mask(f::FESpaceWithLinearConstraints)
  space = f.space
  ndofs = num_free_dofs(space)
  sdof_to_bdof = setdiff(1:ndofs,f.mDOF_to_DOF)
  get_dof_to_mask(sdof_to_bdof,ndofs)
end

function get_dof_to_mask(f::FESpaceWithConstantFixed)
  space = f.space
  ndofs = num_free_dofs(space)
  get_dof_to_mask(f.dof_to_fix,ndofs)
end

function get_dof_to_mask(f::ZeroMeanFESpace)
  space = f.space
  get_dof_to_mask(space)
end

function get_dof_to_mask(masked_dofs,ndofs::Int)
  dof_to_mask = fill(false,ndofs)
  for dof in masked_dofs
    dof_to_mask[dof] = true
  end
  return dof_to_mask
end

"""
    get_polynomial_order(fs::FESpace) -> Integer

Retrieves the polynomial order of `fs`
"""
get_polynomial_order(fs::SingleFieldFESpace) = get_polynomial_order(get_fe_basis(fs))
get_polynomial_order(fs::MultiFieldFESpace) = maximum(map(get_polynomial_order,fs.spaces))

function get_polynomial_order(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_order(shapefun.fields)
end

"""
    get_polynomial_orders(fs::FESpace) -> Integer

Retrieves the polynomial order of `fs` for every dimension
"""
get_polynomial_orders(fs::SingleFieldFESpace) = get_polynomial_orders(get_fe_basis(fs))
get_polynomial_orders(fs::MultiFieldFESpace) = maximum.(map(get_polynomial_orders,fs.spaces))

function get_polynomial_orders(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_orders(shapefun.fields)
end
