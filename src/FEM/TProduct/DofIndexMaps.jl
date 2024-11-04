"""
    get_dof_index_map(space::FESpace) -> AbstractIndexMap

Returns the dofs sorted by coordinate order, for every dimension. Therefore,
if `space` is a D-dimensional FESpace, the output index map will be a subtype
of AbstractIndexMap{D}.

The following example clarifies the function's task:

cell_dof_ids = Table([
  [1, 3, 9, 7],
  [3, 2, 8, 9],
  [7, 9, 6, 4],
  [9, 8, 6, 5]
])

  get_dof_index_map(⋅)
      ⟹ ⟹ ⟹

      [ 1  7  4
        3  9  6
        2  8  5 ]

"""
function get_dof_index_map(space::FESpace)
  trian = get_triangulation(space)
  @notimplementedif !isa(trian,BodyFittedTriangulation)
  trian_ids = trian.tface_to_mface
  model = get_background_model(trian)
  index_map = get_dof_index_map(model,space,trian_ids)
  return remove_dirichlet_dofs(index_map)
end

function get_dof_index_map(model::DiscreteModel,space::FESpace,args...)
  TrivialIndexMap(num_free_dofs(space))
end

function get_dof_index_map(model::CartesianDiscreteModel,space::FESpace,args...)
  @abstractmethod
end

function get_dof_index_map(
  model::CartesianDiscreteModel,
  space::UnconstrainedFESpace,
  trian_ids::AbstractVector)

  dof = get_fe_dof_basis(space)
  T = get_dof_type(dof)
  cell_dof_ids = get_cell_dof_ids(space)
  order = get_polynomial_order(space)
  comp_to_dofs = get_comp_to_dofs(T,space,dof)
  get_dof_index_map(T,model,cell_dof_ids,trian_ids,order,comp_to_dofs)
end

function get_dof_index_map(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  cell_dof_ids::Table{Ti},
  trian_ids::AbstractVector,
  order::Integer,
  args...
  ) where {T,Ti,D}

  dof_map = _get_dof_index_map(model,cell_dof_ids,trian_ids,order)
  return IndexMap(dof_map)
end

function get_dof_index_map(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  cell_dof_ids::Table{Ti},
  trian_ids::AbstractVector,
  order::Integer,
  comp_to_dofs::AbstractVector
  ) where {T<:MultiValue,Ti,D}

  dof_perms = Array{Ti,D}[]
  for dofs in comp_to_dofs
    cell_dof_comp_ids = _get_cell_dof_comp_ids(cell_dof_ids,dofs)
    dof_perm_comp = _get_dof_index_map(model,cell_dof_comp_ids,trian_ids,order)
    push!(dof_perms,dof_perm_comp)
  end
  return MultiValueIndexMap(dof_perms)
end

function _get_dof_index_map(
  model::CartesianDiscreteModel{Dc},
  cell_dof_ids::Table{Ti},
  trian_ids::AbstractVector,
  order::Integer) where {Dc,Ti}

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  new_dof_ids = LinearIndices(ndofs)
  dof_map = fill(zero(Ti),ndofs)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    if icell ∈ trian_ids
      icell′ = findfirst(trian_ids.==icell)
      first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
      new_dofs_range = map(i -> i:i+order,first_new_dof)
      new_dofs = view(new_dof_ids,new_dofs_range...)
      cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell′)
      for (idof,dof) in enumerate(cell_dofs)
        t = terms[idof]
        # new_dofs[t] < 0 && continue
        dof < 0 && continue
        dof_map[new_dofs[t]] = dof
      end
    end
  end

  return dof_map
end

# spaces with constraints

function get_dof_index_map(model::CartesianDiscreteModel,ls::FESpaceWithLinearConstraints,args...)
  space = ls.space
  mdof_to_bdof = ls.mDOF_to_DOF
  sdof_to_bdof = setdiff(1:ls.n_fdofs,mdof_to_bdof)
  index_map = get_dof_index_map(model,space,args...)
  return ConstrainedDofsIndexMap(index_map,mdof_to_bdof,sdof_to_bdof)
end

function get_dof_index_map(model::CartesianDiscreteModel,cs::FESpaceWithConstantFixed,args...)
  space = cs.space
  ndofs = num_free_dofs(space) + num_dirichlet_dofs(space)
  sdof_to_bdof = cs.dof_to_fix
  mdof_to_bdof = setdiff(1:ndofs,sdof_to_bdof)
  index_map = get_dof_index_map(model,space,args...)
  return ConstrainedDofsIndexMap(index_map,mdof_to_bdof,sdof_to_bdof)
end

function get_dof_index_map(model::CartesianDiscreteModel,zs::ZeroMeanFESpace,args...)
  space = zs.space
  get_dof_index_map(model,space,args...)
end

# utils

"""
    get_polynomial_order(fs::FESpace) -> Integer

Retrieves the polynomial order of `fs`

"""
get_polynomial_order(fs::SingleFieldFESpace) = get_polynomial_order(get_fe_basis(fs))
get_polynomial_order(fs::MultiFieldFESpace) = maximum(map(get_polynomial_order,fs.spaces))

function get_polynomial_order(basis)
  cell_basis = get_data(basis)
  shapefun = first(cell_basis).fields
  get_order(shapefun)
end

get_dof_type(b) = @abstractmethod
get_dof_type(b::LagrangianDofBasis{P,V}) where {P,V} = change_eltype(V,Float64)
get_dof_type(dof::CellDof) = get_dof_type(first(get_data(dof)))

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
  b = first(get_data(dof))
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
