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

function get_bg_dof_to_act_dof(f::FESpace)
  @abstractmethod
end

function get_bg_dof_to_mask(f::FESpace)
  map(iszero,get_bg_dof_to_act_dof(f))
end

function get_bg_dof_to_act_dof(f::SingleFieldFESpace)
  IdentityVector(num_free_dofs(f))
end

function get_bg_dof_to_act_dof(f::FESpaceWithLinearConstraints)
  bg_space = f.space
  ndofs = num_free_dofs(bg_space)
  sdof_to_bdof = setdiff(1:ndofs,f.mDOF_to_DOF)
  get_bg_dof_to_act_dof(sdof_to_bdof,ndofs)
end

function get_bg_dof_to_act_dof(f::FESpaceWithConstantFixed)
  bg_space = f.space
  ndofs = num_free_dofs(bg_space)
  get_bg_dof_to_act_dof(f.dof_to_fix,ndofs)
end

function get_bg_dof_to_act_dof(f::ZeroMeanFESpace)
  bg_space = f.space
  get_bg_dof_to_act_dof(bg_space)
end

function get_bg_dof_to_act_dof(masked_dofs,ndofs::Int)
  bg_dof_to_act_dof = collect(1:ndofs)
  for dof in masked_dofs
    bg_dof_to_act_dof[dof] = 0
  end
  return bg_dof_to_act_dof
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
