"""
    get_dof_map(space::FESpace) -> AbstractDofMap

Returns the dofs sorted by coordinate order, for every dimension. Therefore,
if `space` is a D-dimensional FESpace, the output index map will be a subtype
of AbstractDofMap{D}.

The following example clarifies the function's task:

cell_dof_ids = Table([
  [1, 3, 9, 7],
  [3, 2, 8, 9],
  [7, 9, 6, 4],
  [9, 8, 6, 5]
])

  get_dof_map(⋅)
      ⟹ ⟹ ⟹

      [ 1  7  4
        3  9  6
        2  8  5 ]

"""
function get_dof_map(space::FESpace)
  model = get_background_model(trian)
  get_dof_map(model,space)
end

function get_dof_map(test::SingleFieldFESpace)
  TrivialDofMap(num_free_dofs(test))
end

function get_dof_map(test::MultiFieldFESpace)
  ntest = num_fields(test)
  map(1:ntest) do i
    get_dof_map(test[i])
  end
end

function get_dof_map(model::DiscreteModel,space::FESpace)
  @abstractmethod
end

function get_dof_map(model::CartesianDiscreteModel,space::UnconstrainedFESpace)
  cell_dof_ids = get_cell_dof_ids(space)
  dof = get_fe_dof_basis(space)
  T = get_dof_type(dof)
  order = get_polynomial_order(space)
  comp_to_dofs = get_comp_to_dofs(T,space,dof)

  dof_map,dof_owner = get_dof_map(T,model,cell_dof_ids,order,comp_to_dofs)

  trian = get_triangulation(space)
  cell_to_mask = get_cell_to_mask(trian)

  return DofMap(dof_map,dof_owner,cell_to_mask)
end

function get_dof_map(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  cell_dof_ids::Table{Ti},
  order::Integer,
  args...
  ) where {T,Ti,D}

  _get_dof_dof_map(model,cell_dof_ids,order)
end

function get_dof_map(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  cell_dof_ids::Table{Ti},
  order::Integer,
  comp_to_dofs::AbstractVector
  ) where {T<:MultiValue,Ti,D}

  dof_maps = Array{Ti,D}[]
  dof_owners = Array{Ti,D}[]
  for dofs in comp_to_dofs
    cell_dof_comp_ids = _get_cell_dof_comp_ids(cell_dof_ids,dofs)
    dof_map_comp,dof_owner_comp = _get_dof_dof_map(model,cell_dof_comp_ids,order)
    push!(dof_maps,dof_map_comp)
    push!(dof_owners,dof_owner_comp)
  end
  dof_map = stack(dof_maps;dims=D+1)
  dof_owner = stack(dof_maps;dims=D+1)
  return dof_map,dof_owner
end

function _get_dof_dof_map(
  model::CartesianDiscreteModel{Dc},
  cell_dof_ids::Table{Ti},
  order::Integer) where {Dc,Ti}

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  ordered_dof_ids = LinearIndices(ndofs)
  dof_map = zeros(Ti,ndofs)
  dof_to_first_owner_cell = zeros(Ti,ndofs)
  touched_dof = zeros(Bool,ndofs)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    ordered_dofs_range = map(i -> i:i+order,first_new_dof)
    ordered_dofs = view(ordered_dof_ids,ordered_dofs_range...)
    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      odof = ordered_dofs[t]
      (dof < 0 || touched_dof[odof]) && continue
      touched_dof[odof] = true
      dof_map[odof] = dof
      dof_to_first_owner_cell[odof] = icell
    end
  end

  return dof_map,dof_to_first_owner_cell
end

# spaces with constraints

function get_dof_map(model::CartesianDiscreteModel,ls::FESpaceWithLinearConstraints)
  space = ls.space
  ndofs = num_free_dofs(space)
  sdof_to_bdof = setdiff(1:ndofs,ls.mDOF_to_DOF)
  dof_to_constraints = get_dof_to_constraints(sdof_to_bdof,ndofs)
  dof_map = get_dof_map(model,space)
  return ConstrainedDofsDofMap(dof_map,dof_to_constraints)
end

function get_dof_map(model::CartesianDiscreteModel,cs::FESpaceWithConstantFixed)
  space = cs.space
  ndofs = num_free_dofs(space)
  dof_to_constraints = get_dof_to_constraints(cs.sdof_to_bdof,ndofs)
  dof_map = get_dof_map(model,space)
  return ConstrainedDofsDofMap(dof_map,dof_to_constraints)
end

function get_dof_map(model::CartesianDiscreteModel,zs::ZeroMeanFESpace)
  space = zs.space
  get_dof_map(model,space)
end

# sparse interface

"""
    get_sparse_dof_map(trial::FESpace,test::FESpace) -> AbstractDofMap

Returns the index maps related to jacobians in a FE problem. The default output
is a TrivialDofMap; when the trial and test spaces are of type TProductFESpace,
a SparseDofMap is returned

"""
function get_sparse_dof_map(trial::SingleFieldFESpace,test::SingleFieldFESpace)
  sparsity = SparsityPattern(trial,test)
  get_sparse_dof_map(trial,test,sparsity)
end

function get_sparse_dof_map(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  map(Iterators.product(1:ntest,1:ntrial)) do i,j
    get_sparse_dof_map(trial[j],test[i])
  end
end

function get_sparse_dof_map(
  trial::SingleFieldFESpace,
  test::SingleFieldFESpace,
  sparsity::TProductSparsityPattern)

  osparsity = order_sparsity(sparsity,trial,test)
  I,J,V = findnz(osparsity)
  i,j,v = univariate_findnz(osparsity)
  sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
  # osparse_indices = order_dof_map(sparse_indices,trial,test)
  full_indices = to_nz_index(sparse_indices,sparsity)
  SparseDofMap(pg2l,pg2l_sparse,osparsity)
end

function get_sparse_dof_map(
  trial::SingleFieldFESpace,
  test::SingleFieldFESpace,
  sparsity::SparsityPattern)

  TrivialDofMap(sparsity)
end

# utils

function get_cell_to_mask(t::Triangulation)
  tface_to_mface = get_tface_to_mface(t)
  model = get_background_model(t)
  ncells = num_cells(model)
  isa(tface_to_mface,IdentityVector) && return Fill(true,ncells)
  cell_to_mask = fill(true,ncells)
  for cell in eachindex(cell_to_mask)
    if !(cell ∈ tface_to_mface)
      cell_to_mask[cell] = false
    end
  end
  return cell_to_mask
end

function get_dof_to_constraints(constrained_dofs::AbstractVector,ndofs::Int)
  dof_to_constraint = fill(true,ndofs)
  for dof in eachindex(dof_to_constraint)
    if !(dof ∈ constrained_dofs)
      dof_to_constraint[dof] = false
    end
  end
  return dof_to_constraint
end

get_tface_to_mface(t::Geometry.BodyFittedTriangulation) = t.tface_to_mface
get_tface_to_mface(t::Geometry.BoundaryTriangulation) = t.glue.face_to_bgface
get_tface_to_mface(t::Geometry.TriangulationView) = get_tface_to_mface(t.parent)
get_tface_to_mface(t::Interfaces.SubFacetTriangulation) = t.subfacets.facet_to_bgcell
get_tface_to_mface(t::Interfaces.SubCellTriangulation) = t.subcells.cell_to_bgcell
function get_tface_to_mface(t::Geometry.AppendedTriangulation)
  lazy_append(get_tface_to_mface(t.a),get_tface_to_mface(t.b))
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

get_dof_type(b) = @abstractmethod
get_dof_type(b::LagrangianDofBasis{P,V}) where {P,V} = change_eltype(V,Float64)
get_dof_type(dof::CellDof) = get_dof_type(testitem(get_data(dof)))

function get_comp_to_dofs(::Type{T},space::FESpace,dof::CellDof) where T
  @abstractmethod
end

function get_comp_to_dofs(::Type{T},space::UnconstrainedFESpace,dof::CellDof) where T
  glue = space.metadata
  ncomps = num_components(T)
  dof2comp = if isnothing(glue)
    _get_dofs_to_comp(space,dof)
  else
    vcat(glue.free_dof_to_comp...,glue.dirichlet_dof_to_comp...)
  end
  comp2dof = Vector{typeof(dof2comp)}(undef,ncomps)
  for comp in 1:ncomps
    comp2dof[comp] = findall(dof2comp.==comp)
  end
  return comp2dof
end

function _get_dofs_to_comp(space,dof)
  b = testitem(get_data(dof))
  ldof2comp = get_dof_to_comp(b)
  cell_dof_ids = get_cell_dof_ids(space)
  ndofs = num_free_dofs(space)+num_dirichlet_dofs(space)
  dof2comp = zeros(eltype(ldof2comp),ndofs)
  @inbounds for dofs_cell in cell_dof_ids
    for (ldof,dof) in enumerate(dofs_cell)
      if dof > 0
        dof2comp[dof] = ldof2comp[ldof]
      else
        dof2comp[-dof] = ldof2comp[ldof]
      end
    end
  end
  return dof2comp
end

function _get_terms(p::Polytope,orders)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  terms = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return terms
end

function _get_cell_dof_comp_ids(cell_dof_ids,dofs)
  T = eltype(cell_dof_ids)
  ncells = length(cell_dof_ids)
  new_cell_ids = Vector{T}(undef,ncells)
  cache_cell_dof_ids = array_cache(cell_dof_ids)
  @inbounds for icell in 1:ncells
    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    ids_comp = findall(map(cd->cd ∈ dofs,abs.(cell_dofs)))
    new_cell_ids[icell] = cell_dofs[ids_comp]
  end
  return Table(new_cell_ids)
end

# sparse utilities
# function _permute_dof_map(dof_map,I,J,nrows)
#   IJ = vectorize(I) .+ nrows .* (vectorize(J)'.-1)
#   iperm = copy(dof_map)
#   @inbounds for (k,pk) in enumerate(dof_map)
#     if pk > 0
#       iperm[k] = IJ[pk]
#     end
#   end
#   return iperm
# end

# function _permute_dof_map(dof_map,I::AbstractMultiValueDofMap,J::AbstractMultiValueDofMap,nrows)
#   function _to_component_indices(i,ncomps,icomp_I,icomp_J,nrows)
#     ic = copy(i)
#     @inbounds for (j,IJ) in enumerate(i)
#       IJ == 0 && continue
#       I = fast_index(IJ,nrows)
#       J = slow_index(IJ,nrows)
#       I′ = (I-1)*ncomps + icomp_I
#       J′ = (J-1)*ncomps + icomp_J
#       ic[j] = (J′-1)*nrows*ncomps + I′
#     end
#     return ic
#   end

#   ncomps_I = num_components(I)
#   ncomps_J = num_components(J)
#   @check ncomps_I == ncomps_J
#   ncomps = ncomps_I
#   nrows_per_comp = Int(nrows/ncomps)

#   I1 = get_component(I,1;multivalue=false)
#   J1 = get_component(J,1;multivalue=false)
#   dof_map′ = _permute_dof_map(dof_map,I1,J1,nrows_per_comp)

#   dof_map′′ = map(CartesianIndices((ncomps,ncomps))) do icomp
#     _to_component_indices(dof_map′,ncomps,icomp.I[1],icomp.I[2],nrows_per_comp)
#   end
#   return MultiValueDofMap(dof_map′′)
# end

# function _permute_dof_map(dof_map,I::AbstractMultiValueDofMap,J::AbstractDofMap,nrows)
#   function _to_component_indices(i,ncomps,icomp,nrows)
#     ic = copy(i)
#     @inbounds for (j,IJ) in enumerate(i)
#       IJ == 0 && continue
#       I = fast_index(IJ,nrows)
#       J = slow_index(IJ,nrows)
#       I′ = (I-1)*ncomps + icomp
#       ic[j] = (J-1)*nrows*ncomps + I′
#     end
#     return ic
#   end

#   ncomps = num_components(I)
#   nrows_per_comp = Int(nrows/ncomps)

#   I1 = get_component(I,1;multivalue=false)
#   dof_map′ = _permute_dof_map(dof_map,I1,J,nrows_per_comp)

#   dof_map′′ = map(icomp->_to_component_indices(dof_map′,ncomps,icomp,nrows_per_comp),1:ncomps)
#   return MultiValueDofMap(dof_map′′)
# end

# function _permute_dof_map(dof_map,I::AbstractDofMap,J::AbstractMultiValueDofMap,nrows)
#   function _to_component_indices(i,ncomps,icomp,nrows)
#     ic = copy(i)
#     @inbounds for (j,IJ) in enumerate(i)
#       IJ == 0 && continue
#       I = fast_index(IJ,nrows)
#       J = slow_index(IJ,nrows)
#       J′ = (J-1)*ncomps + icomp
#       ic[j] = (J′-1)*nrows + I
#     end
#     return ic
#   end

#   ncomps = num_components(J)

#   J1 = get_component(J,1;multivalue=false)
#   dof_map′ = _permute_dof_map(dof_map,I,J1,nrows)
#   dof_map′′ = map(icomp->_to_component_indices(dof_map′,ncomps,icomp,nrows),1:ncomps)
#   return MultiValueDofMap(dof_map′′)
# end

# function _permute_dof_map(dof_map,trial::TProductFESpace,test::TProductFESpace)
#   I = get_dof_map(test)
#   J = get_dof_map(trial)
#   nrows = num_free_dofs(test)
#   return _permute_dof_map(dof_map,I,J,nrows)
# end
