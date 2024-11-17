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

function get_dof_map(space::SingleFieldFESpace)
  trian = get_triangulation(space)
  nfdofs = num_free_dofs(space)
  cell_dof_ids = get_cell_dof_ids(space)
  cache = array_cache(cell_dof_ids)

  dof_to_cell = _get_dof_to_cell(cache,cell_dof_ids,nfdofs)
  cell_to_mask = get_cell_to_mask(trian)
  TrivialDofMap(dof_to_cell,cell_to_mask)
end

function get_dof_map(space::MultiFieldFESpace)
  nfields = num_fields(space)
  map(1:nfields) do i
    get_dof_map(space[i])
  end
end

function get_dof_map(space::FESpace,trian::Union{Triangulation,Tuple{Vararg{Triangulation}}})
  dof_map = get_dof_map(space)
  change_domain(dof_map,trian)
end

function CellData.change_domain(dof_map::AbstractDofMap,trian::Triangulation)
  @abstractmethod
end

function CellData.change_domain(dof_maps::AbstractArray{<:AbstractDofMap},trian::Triangulation)
  map(dof_maps) do dof_map
    change_domain(dof_map,trian)
  end
end

function CellData.change_domain(dof_map,trians::Tuple{Vararg{Triangulation}})
  contribution(trians) do trian
    change_domain(dof_map,trian)
  end
end

function get_dof_map(model::DiscreteModel,space::FESpace)
  @abstractmethod
end

function get_dof_map(model::CartesianDiscreteModel,space::UnconstrainedFESpace)
  cell_dof_ids = get_cell_dof_ids(space)
  cache = array_cache(cell_dof_ids)
  dof = get_fe_dof_basis(space)
  T = get_dof_type(dof)
  order = get_polynomial_order(space)
  comp_to_dofs = get_comp_to_dofs(T,space,dof)

  dof_map = get_dof_map(T,model,cache,cell_dof_ids,order,comp_to_dofs)

  trian = get_triangulation(space)
  cell_to_mask = get_cell_to_mask(trian)

  nfdofs = num_free_dofs(space)
  dof_to_cell = _get_dof_to_cell(cache,cell_dof_ids,nfdofs)

  return DofMap(dof_map,dof_to_cell,cell_to_mask)
end

function get_dof_map(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  cache,
  cell_dof_ids::Table{Ti},
  order::Integer,
  args...
  ) where {T,Ti,D}

  get_dof_map(model,cache,cell_dof_ids,order)
end

function get_dof_map(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  cache,
  cell_dof_ids::Table{Ti},
  order::Integer,
  comp_to_dofs::AbstractVector
  ) where {T<:MultiValue,Ti,D}

  dof_maps = Array{Ti,D}[]
  for dofs in comp_to_dofs
    cell_dof_comp_ids = _get_cell_dof_comp_ids(cell_dof_ids,dofs)
    dof_map_comp = get_dof_map(model,cache,cell_dof_comp_ids,order)
    push!(dof_maps,dof_map_comp)
  end
  dof_map = stack(dof_maps;dims=D+1)
  return dof_map
end

function get_dof_map(
  model::CartesianDiscreteModel{Dc},
  cache_cell_dof_ids,
  cell_dof_ids::Table{Ti},
  order::Integer) where {Dc,Ti}

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))

  ordered_dof_ids = LinearIndices(ndofs)
  dof_map = zeros(Ti,ndofs)
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
    end
  end

  return dof_map
end

function _get_dof_to_cell(cache,cell_dof_ids,nfdofs)
  dof_to_cell = map(dof -> get_dof_to_cell(cache,cell_dof_ids,dof),1:nfdofs)
  return Table(dof_to_cell)
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

get_univariate_dof_map(f::SingleFieldFESpace) = @abstractmethod
get_univariate_dof_map(f::MultiFieldFESpace) = @notimplemented

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
  map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    get_sparse_dof_map(trial[j],test[i])
  end
end

function get_sparse_dof_map(
  trian::Triangulation{D},
  rows::AbstractDofMap{D},
  cols::AbstractDofMap{D},
  unrows::AbstractVector,
  uncols::AbstractVector,
  sparsity::TProductSparsityPattern
  ) where D

  osparsity = order_sparsity(sparsity,(rows,unrows),(cols,uncols))
  I,J,V = findnz(osparsity)
  i,j,v = univariate_findnz(osparsity)
  sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
  osparse_indices = order_sparse_dof_map(sparse_indices,rows,cols)
  return osparsity,osparse_indices
end

function get_sparse_dof_map(
  trian::Triangulation{D},
  rows::AbstractDofMap{Dr},
  cols::AbstractDofMap{Dc},
  unrows::AbstractVector,
  uncols::AbstractVector,
  sparsity::TProductSparsityPattern
  ) where {D,Dr,Dc}

  @check Dr ≥ D
  @check Dc ≥ D

  first_component(a::AbstractDofMap{D};kwargs...) = a
  first_component(a::AbstractDofMap{D′};kwargs...) where D′ = get_component(a;kwargs...)
  ncomponents(a::AbstractDofMap{D}) = 1
  ncomponents(a::AbstractDofMap{D′}) where D′ = size(a,D′)

  rows′ = first_component(rows;to_scalar=false)
  cols′ = first_component(cols;to_scalar=false)
  rows′′ = first_component(rows;to_scalar=true)
  cols′′ = first_component(cols;to_scalar=true)
  ncomps_row = ncomponents(rows)
  ncomps_col = ncomponents(cols)

  osparsity = order_sparsity(sparsity,(rows′,unrows),(cols′,uncols))
  I,J,V = findnz(osparsity)
  i,j,v = univariate_findnz(osparsity)
  sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
  osparse_indices = order_sparse_dof_map(sparse_indices,rows′′,cols′′)

  osparse_indices_comp = add_sparse_components(osparse_indices,rows′′,cols′′,ncomps_row,ncomps_col)
  return osparsity,osparse_indices_comp
end

function get_sparse_dof_map(
  trial::SingleFieldFESpace,
  test::SingleFieldFESpace,
  sparsity::TProductSparsityPattern)

  trian = get_triangulation(trial)
  model = get_background_model(trian)
  @check model === get_background_model(get_triangulation(test))

  rows = get_dof_map(test)
  cols = get_dof_map(trial)
  unrows = get_univariate_dof_map(test)
  uncols = get_univariate_dof_map(trial)

  osparsity,osparse_indices = get_sparse_dof_map(trian,rows,cols,unrows,uncols,sparsity)
  ofull_indices = to_nz_index(osparse_indices,sparsity)

  SparseDofMap(ofull_indices,osparse_indices,osparsity)
end

function get_sparse_dof_map(
  trial::SingleFieldFESpace,
  test::SingleFieldFESpace,
  sparsity::SparsityPattern)

  TrivialSparseDofMap(sparsity)
end

function get_sparse_dof_map(
  trial::FESpace,
  test::FESpace,
  trian::Union{Triangulation,Tuple{Vararg{Triangulation}}})

  sparse_dof_map = get_sparse_dof_map(trial,test)
  rows = get_dof_map(test,trian)
  cols = get_dof_map(trial,trian)
  change_domain(sparse_dof_map,rows,cols)
end

function CellData.change_domain(
  sparse_dof_map::AbstractDofMap,
  rows::AbstractDofMap,
  cols::AbstractDofMap)

  @abstractmethod
end

function CellData.change_domain(
  sparse_dof_maps::AbstractMatrix{<:AbstractDofMap},
  rows::AbstractVector{<:AbstractDofMap},
  cols::AbstractVector{<:AbstractDofMap})

  @check size(sparse_dof_maps,1) == length(rows)
  @check size(sparse_dof_maps,2) == length(cols)
  map(Iterators.product(1:length(rows),1:length(cols))) do (i,j)
    change_domain(sparse_dof_maps[i,j],rows[i],cols[j])
  end
end

function CellData.change_domain(
  sparse_dof_map,
  rows::ArrayContribution,
  cols::ArrayContribution)

  @check all( ( tr === tc for (tr,tc) in zip(get_domains(rows),get_domains(cols)) ) )
  trians = get_domains(rows)
  contribution(trians) do trian
    change_domain(sparse_dof_map,rows[trian],cols[trian])
  end
end

# utils

function get_cell_to_mask(t::Triangulation)
  tface_to_mface = get_tface_to_mface(t)
  model = get_background_model(t)
  ncells = num_cells(model)
  isa(tface_to_mface,IdentityVector) && return Fill(false,ncells)
  cell_to_mask = fill(false,ncells)
  for cell in eachindex(cell_to_mask)
    if !(cell ∈ tface_to_mface)
      cell_to_mask[cell] = true
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
get_tface_to_mface(t::Interfaces.SubCellTriangulation) = unique(t.subcells.cell_to_bgcell)
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

function get_dof_to_cell(cache,cell_dof_ids,dof)
  cells = Int32[]
  for cell in 1:length(cell_dof_ids)
    cell_dofs = getindex!(cache,cell_dof_ids,cell)
    if dof ∈ cell_dofs
      append!(cells,cell)
    end
  end
  cells
end

# sparse utilities

function order_sparse_dof_map(dof_map,I,J,nrows=maximum(I))
  IJ = vectorize(I) .+ nrows .* (vectorize(J)'.-1)
  odof_map = copy(dof_map)
  @inbounds for (k,dofk) in enumerate(dof_map)
    if dofk > 0
      odof_map[k] = IJ[dofk]
    end
  end
  return odof_map
end

# multi component

function add_sparse_components(dof_map,I,J,ncomps_I,ncomps_J,nrows=maximum(I))
  Ti = eltype(dof_map)
  ncomps_IJ = ncomps_I*ncomps_J
  dof_maps = zeros(Ti,size(dof_map)...,ncomps_IJ)

  for j in CartesianIndices(dof_map)
    IJ = dof_map[j]
    IJ == 0 && continue
    I = fast_index(IJ,nrows)
    J = slow_index(IJ,nrows)
    for icomp_J in 1:ncomps_J
      for icomp_I in 1:ncomps_I
        icomp_IJ = icomp_I + ncomps_I*(icomp_J-1)
        I′ = (I-1)*ncomps_I + icomp_I
        J′ = (J-1)*ncomps_J + icomp_J
        dof_maps[j,icomp_IJ] = (J′-1)*nrows*ncomps_I + I′
      end
    end
  end

  return dof_maps
end
