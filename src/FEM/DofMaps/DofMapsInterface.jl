"""
    abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

Type representing an ordering of dofs in a certain `FESpace`. This type
is used for efficient (lazy) reindexing of the free values of a `FEFunction`
defined on the aforementioned FESpace. Subtypes:

- `AbstractTrivialDofMap`
- `DofMap`
- `ConstrainedDofMap`
- `DofMapPortion`
- `TProductDofMap`
- `ConstrainedTProductDofMap`
- `SparseDofMap`
"""
abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

Base.IndexStyle(i::AbstractDofMap) = IndexLinear()

FESpaces.ConstraintStyle(i::I) where I<:AbstractDofMap = ConstraintStyle(I)
FESpaces.ConstraintStyle(::Type{<:AbstractDofMap}) = UnConstrained()

get_dof_to_cell(i::AbstractDofMap) = @abstractmethod
get_tface_to_mask(i::AbstractDofMap) = @abstractmethod
Utils.get_tface_to_mface(i::AbstractDofMap) = @abstractmethod

function remove_constrained_dofs(i::AbstractDofMap)
  remove_constrained_dofs(i,ConstraintStyle(i))
end

function remove_constrained_dofs(i::AbstractDofMap,::UnConstrained)
  collect(i)
end

function remove_constrained_dofs(i::AbstractDofMap{Ti},::Constrained) where Ti
  i′ = collect(vec(i))
  deleteat!(i′,get_dof_to_constraints(i))
  i′
end

get_dof_to_constraints(i::AbstractDofMap) = @abstractmethod

TensorValues.num_components(::Type{<:AbstractDofMap{D,Ti}}) where {D,Ti} = num_components(Ti)

"""
    vectorize(i::AbstractDofMap) -> AbstractVector

Reshapes `i` as a vector, and removes the constrained dofs in `i` (if any)
"""
function vectorize(i::AbstractDofMap)
  vec(remove_constrained_dofs(i))
end

"""
    invert(i::AbstractDofMap;kwargs...) -> AbstractArray

Calls the function `invperm` on the nonzero entries of `i`, and places
zeros on the remaining zero entries of `i`. The output has the same size as `i`
"""
function invert(i::AbstractDofMap;kwargs...)
  invert(i,ConstraintStyle(i);kwargs...)
end

function invert(i::AbstractArray,args...)
  i′ = similar(i)
  invert!(i′,i)
  i′
end

function invert!(i′::AbstractArray,i::AbstractArray)
  fill!(i′,zero(eltype(i′)))
  inz = findall(!iszero,i)
  i′[inz] = invperm(i[inz])
end

function invert(i::AbstractDofMap,::Constrained)
  i′ = remove_constrained_dofs(i)
  invert(i′)
end

# i1 ∘ i2
function compose_maps(i1::AbstractArray{Ti,D},i2::AbstractArray{Ti,D}) where {Ti,D}
  @assert size(i1) == size(i2)
  i12 = zeros(Ti,size(i1))
  for (i,m2i) in enumerate(i2)
    iszero(m2i) && continue
    i12[i] = i1[m2i]
  end
  return i12
end

function get_dof_to_parent_dof_map(i::AbstractDofMap,parent::AbstractDofMap)
  @abstractmethod
end

# returns the dof `i[idof]` if `idof` is not masked, and zero otherwise
@inline function show_dof(i::AbstractDofMap,idof::Integer)
  iszero(idof) && return true
  dof_to_cell = get_dof_to_cell(i)
  tface_to_mask = get_tface_to_mask(i)

  pini = dof_to_cell.ptrs[idof]
  pend = dof_to_cell.ptrs[idof+1]-1

  show = false
  for j in pini:pend
    if !tface_to_mask[dof_to_cell.data[j]]
      show = true
      break
    end
  end

  return show
end

"""
    abstract type AbstractTrivialDofMap{Ti} <: AbstractDofMap{1,Ti} end

Type representing the trivial dof maps, which are returned when we do not desire
a reindexing of the entries. Subtypes:

- `TrivialDofMap`
- `TrivialSparseDofMap`
"""
abstract type AbstractTrivialDofMap{Ti} <: AbstractDofMap{1,Ti} end

Base.setindex!(i::AbstractTrivialDofMap,v,j::Integer) = v
Base.copy(i::AbstractTrivialDofMap) = i
Base.similar(i::AbstractTrivialDofMap) = i

"""
    struct TrivialDofMap{Ti,A<:AbstractVector,B<:AbstractVector} <: AbstractTrivialDofMap{Ti}
      dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
      tface_to_mask::A
      tface_to_mface::B
    end

In its simples case, it should be conceived as a unit-range from 1 to the number
of free dofs of a given FE space. The fields `dof_to_cell`, `tface_to_mask` and
`tface_to_mface` are provided in order to successfully change the underlying
triangulation via `change_domain`, if needed
"""
struct TrivialDofMap{Ti,A<:AbstractVector,B<:AbstractVector} <: AbstractTrivialDofMap{Ti}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  tface_to_mask::A
  tface_to_mface::B
end

function TrivialDofMap(i::AbstractDofMap)
  TrivialDofMap(get_dof_to_cell(i),get_tface_to_mask(i),get_tface_to_mface(i))
end

Base.size(i::TrivialDofMap) = (length(i.dof_to_cell),)

Base.@propagate_inbounds function Base.getindex(i::TrivialDofMap{Ti},j::Integer) where Ti
  show_dof(i,j) ? j : zero(Ti)
end

get_dof_to_cell(i::TrivialDofMap) = i.dof_to_cell
get_tface_to_mask(i::TrivialDofMap) = i.tface_to_mask
Utils.get_tface_to_mface(i::TrivialDofMap) = i.tface_to_mface

function CellData.change_domain(i::TrivialDofMap,t::Triangulation)
  tface_to_mface = get_tface_to_mface(t)
  tface_to_mask = get_tface_to_mask(tface_to_mface,i.tface_to_mface)
  TrivialDofMap(i.dof_to_cell,tface_to_mask,i.tface_to_mface)
end

"""
    struct DofMap{D,Ti,A<:AbstractVector,B<:AbstractVector} <: AbstractDofMap{D,Ti}
      indices::Array{Ti,D}
      dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
      free_vals_box::Array{Int,D}
      tface_to_mask::A
      tface_to_mface::B
    end

Fields:
- `indices`: D-array of indices corresponding to the reordered dofs
- `dof_to_cell`: the inverse function of cell_dof_ids in `Gridap`, which associates
  a list of dofs to a given cell
- `free_vals_box`: D-array containing the indices of the free dofs in `indices`
- `tface_to_mask`: list returned by `get_tface_to_mask`. By default, this
  field considers the `tface_to_mask` associated to the triangulation of the FE
  space. However, when changing the triangulation of the index map via
  `change_domain`, we change the `tface_to_mask`
- `tface_to_mface`: list of active cells of the triangulation of the FE space we
  consider
"""
struct DofMap{D,Ti,A<:AbstractVector,B<:AbstractVector} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  free_vals_box::Array{Int,D}
  tface_to_mask::A
  tface_to_mface::B
end

function DofMap(
  indices::AbstractArray,
  dof_to_cell::Table,
  tface_to_mask::AbstractVector,
  tface_to_mface::AbstractVector,
  diri_entities::AbstractMatrix)

  free_vals_box = find_free_values_box(indices,diri_entities)
  DofMap(indices,dof_to_cell,free_vals_box,tface_to_mask,tface_to_mface)
end

Base.size(i::DofMap) = size(i.free_vals_box)

Base.@propagate_inbounds function Base.getindex(
  i::DofMap{D,Ti},
  j::Integer
  ) where {D,Ti}

  free_j = i.free_vals_box[j]
  dof_j = i.indices[free_j]
  show_dof(i,dof_j) ? dof_j : zero(Ti)
end

Base.@propagate_inbounds function Base.setindex!(
  i::DofMap{D,Ti},
  v,j::Integer
  ) where {D,Ti}

  free_j = i.free_vals_box[j]
  dof_j = i.indices[free_j]
  if show_dof(i,dof_j)
    i.indices[free_j] = v
  end
  v
end

function Base.copy(i::DofMap)
  DofMap(copy(i.indices),i.dof_to_cell,i.free_vals_box,i.tface_to_mask,i.tface_to_mface)
end

function Base.similar(i::DofMap)
  DofMap(similar(i.indices),i.dof_to_cell,i.free_vals_box,i.tface_to_mask,i.tface_to_mface)
end

function CellData.change_domain(i::DofMap,t::Triangulation)
  tface_to_mface = get_tface_to_mface(t)
  tface_to_mask = get_tface_to_mask(tface_to_mface,i.tface_to_mface)
  DofMap(i.indices,i.dof_to_cell,i.free_vals_box,tface_to_mask,i.tface_to_mface)
end

get_dof_to_cell(i::DofMap) = i.dof_to_cell
get_tface_to_mask(i::DofMap) = i.tface_to_mask
Utils.get_tface_to_mface(i::DofMap) = i.tface_to_mface

function get_dof_to_parent_dof_map(i::DofMap{D,Ti},parent::DofMap{D,Ti}) where {D,Ti}
  face_to_face_parent::Vector{Int} = indexin(i.tface_to_mface,parent.tface_to_mface)
  inv_parent = invert(parent)
  i2parent = compose_maps(i,inv_parent)
  return i2parent
end

# optimization

const EntireDofMap{D,Ti,A<:Fill{Bool},B<:AbstractVector} = DofMap{D,Ti,A,B}

Base.@propagate_inbounds function Base.getindex(i::EntireDofMap,j::Integer)
  free_j = i.free_vals_box[j]
  i.indices[free_j]
end

Base.@propagate_inbounds function Base.setindex!(i::EntireDofMap,v,j::Integer)
  free_j = i.free_vals_box[j]
  i.indices[free_j] = v
  v
end

# multi value

function _to_scalar_values!(i::AbstractArray,D::Integer,d::Integer)
  i .= (i .- d) ./ D .+ 1
end

"""
    get_component(i::AbstractDofMap{D},args...;kwargs...) -> AbstractDofMap{D-1}

Selects the last dimension of `i`, and returns the corresponding dof map. This
function should only be used when the last dimension of `i` corresponds to
the components axis. For example, let us consider a triangulation on a 3D geometry,
on which we define a FE space `f` with a number of dofs per direction (Nx,Ny,Nz).
If `f` is a space of scalar-valued functions, then the size of `i` is (Nx,Ny,Nz),
and calling this function returns a dof map of size (Nx,Ny); if `f` is a space of
vector-valued functions with ncomps components, then the size of `i` is (Nx,Ny,Nz,ncomps),
calling this function returns a dof map of size (Nx,Ny,Nz).

"""
function get_component(i::DofMap{D},d::Integer=1;to_scalar=false) where D
  indices = collect(selectdim(i.indices,D,d))
  free_vals_box = collect(selectdim(i.free_vals_box,D,1))
  dof_to_cell = get_dof_to_cell(i)

  if to_scalar
    ncomps = size(i,D)
    δ = ncomps-d
    for j in free_vals_box
      indj = indices[j]
      indices[j] = (indj+δ) / ncomps
    end
    dof_to_cell = dof_to_cell[d:ncomps:length(dof_to_cell)]
  end

  DofMap(indices,dof_to_cell,free_vals_box,i.tface_to_mask,i.tface_to_mface)
end

function get_component(
  trian::Triangulation{Dc,D},
  i::AbstractDofMap{D},
  args...;
  kwargs...
  ) where {Dc,D}

  i
end

function get_component(
  trian::Triangulation{Dc,Dp},
  i::AbstractDofMap{D},
  args...;
  kwargs...
  ) where {Dc,Dp,D}

  get_component(i,args...;kwargs...)
end

function get_component(
  trian::Triangulation,
  i::AbstractTrivialDofMap,
  args...;
  kwargs...
  )

  i
end

function get_component(
  trian::Triangulation,
  i::AbstractVector{<:AbstractDofMap},
  args...;
  kwargs...
  )

  map(i->get_component(trian,i,args...;kwargs...),i)
end

"""
    struct ConstrainedDofMap{D,Ti,A,B} <: AbstractDofMap{D,Ti}
      map::DofMap{D,Ti,A,B}
      dof_to_constraint_mask::Vector{Bool}
    end

Same as a `DofMap`, but contains an additional field `dof_to_constraint_mask`
which tracks the constrained dofs. If a dof is constrained, a zero is shown at its
place
"""
struct ConstrainedDofMap{D,Ti,A,B} <: AbstractDofMap{D,Ti}
  map::DofMap{D,Ti,A,B}
  dof_to_constraint_mask::Vector{Bool}
end

FESpaces.ConstraintStyle(::Type{<:ConstrainedDofMap}) = Constrained()

get_dof_to_cell(i::ConstrainedDofMap) = get_dof_to_cell(i.map)
get_tface_to_mask(i::ConstrainedDofMap) = get_tface_to_mask(i.map)
Utils.get_tface_to_mface(i::ConstrainedDofMap) = get_tface_to_mface(i.map)
get_dof_to_constraints(i::ConstrainedDofMap) = i.dof_to_constraint_mask

Base.size(i::ConstrainedDofMap) = size(i.map)

function Base.getindex(i::ConstrainedDofMap,j::Integer)
  i.dof_to_constraint_mask[j] ? zero(eltype(i)) : getindex(i.map,j)
end

function Base.setindex!(i::ConstrainedDofMap,v,j::Integer)
  !i.dof_to_constraint_mask[j] && setindex!(i.map,v,j)
end

function Base.copy(i::ConstrainedDofMap)
  ConstrainedDofMap(copy(i.map),i.dof_to_constraint_mask)
end

function Base.similar(i::ConstrainedDofMap)
  ConstrainedDofMap(similar(i.map),i.dof_to_constraint_mask)
end

function CellData.change_domain(i::ConstrainedDofMap,t::Triangulation)
  ConstrainedDofMap(change_domain(i.map,t),i.dof_to_constraint_mask)
end

"""
    struct DofMapPortion{D,Ti,A<:AbstractDofMap{D,Ti},B<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
      map::A
      parent_map::B
    end

A dof map for FE spaces defined on portions of a grid, whose information is contained
in the field `map`. The field `parent_map` contains the dof map relative to the
same FE space, when it is defined on the entire underlying grid.
"""
struct DofMapPortion{D,Ti,A<:AbstractDofMap{D,Ti},B<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
  map::A
  parent_map::B
end

FESpaces.ConstraintStyle(::Type{DofMapPortion{D,Ti,I}}) where {D,Ti,I} = ConstraintStyle(I)

get_dof_to_cell(i::DofMapPortion) = get_dof_to_cell(i.map)
get_tface_to_mask(i::DofMapPortion) = get_tface_to_mask(i.map)
Utils.get_tface_to_mface(i::DofMapPortion) = get_tface_to_mface(i.map)

get_parent_map(i::AbstractDofMap) = @abstractmethod
get_parent_map(i::DofMapPortion) = i.parent_map

Base.size(i::DofMapPortion) = size(i.map)

function Base.getindex(i::DofMapPortion,j::Integer)
  getindex(i.map,j)
end

function Base.setindex!(i::DofMapPortion,v,j::Integer)
  setindex!(i.map,v,j)
end

function Base.copy(i::DofMapPortion)
  DofMapPortion(copy(i.map),copy(i.parent_map))
end

function Base.similar(i::DofMapPortion)
  DofMapPortion(similar(i.map),similar(i.parent_map))
end

function CellData.change_domain(i::DofMapPortion,t::Triangulation)
  map′ = change_domain(i.map,t)
  parent_map′ = change_domain(i.parent_map,t)
  DofMapPortion(map′,parent_map′)
end

function get_dof_to_parent_dof_map(i::DofMapPortion)
  get_dof_to_parent_dof_map(i.map,i.parent_map)
end

function get_component(
  trian::Triangulation,
  i::DofMapPortion,
  args...;
  kwargs...
  )

  map′ = get_component(trian,i.map,args...;kwargs...)
  parent_map′ = get_component(trian,i.parent_map,args...;kwargs...)
  DofMapPortion(map′,parent_map′)
end

# tensor product index map, a lot of previous machinery is not needed

"""
    TProductDofMap{D,Ti} <: AbstractDofMap{D,Ti}

Fields:
- `indices`: provides the ordering of dofs for a FE space obtained as the tensor
  product of `D` one-dimensional FE spaces
- `indices_1d`: provides the ordering of dofs for each of the `D` one-dimensional
  FE spaces
"""
struct TProductDofMap{D,Ti} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  indices_1d::Vector{Vector{Ti}}
end

Base.size(i::TProductDofMap) = size(i.indices)

function Base.getindex(i::TProductDofMap,j::Integer)
  getindex(i.indices,j)
end

function Base.setindex!(i::TProductDofMap,v,j::Integer)
  setindex!(i.indices,v,j)
end

function Base.copy(i::TProductDofMap)
  TProductDofMap(copy(i.indices),i.indices_1d,i.dof_map)
end

function Base.similar(i::TProductDofMap)
  TProductDofMap(similar(i.indices),i.indices_1d,i.dof_map)
end

get_univariate_dof_map(i::TProductDofMap) = i.indices_1d

"""
    ConstrainedTProductDofMap{D,Ti} <: AbstractDofMap{D,Ti}

Same as a `TProductDofMap`, but contains an additional field `dof_to_constraint_mask`
which tracks the constrained dofs. If a dof is constrained, a zero is shown at its
place
"""
struct ConstrainedTProductDofMap{D,Ti} <: AbstractDofMap{D,Ti}
  map::TProductDofMap{D,Ti}
  dof_to_constraint_mask::Vector{Bool}
end

FESpaces.ConstraintStyle(::Type{<:ConstrainedTProductDofMap}) = Constrained()

get_dof_to_constraints(i::ConstrainedTProductDofMap) = i.dof_to_constraint_mask

Base.size(i::ConstrainedTProductDofMap) = size(i.map)

function Base.getindex(i::ConstrainedTProductDofMap,j::Integer)
  i.dof_to_constraint_mask[j] ? zero(eltype(i)) : getindex(i.map,j)
end

function Base.setindex!(i::ConstrainedTProductDofMap,v,j::Integer)
  !i.dof_to_constraint_mask[j] && setindex!(i.map,v,j)
end

function Base.copy(i::ConstrainedTProductDofMap)
  ConstrainedTProductDofMap(copy(i.map),i.dof_to_constraint_mask)
end

function Base.similar(i::ConstrainedTProductDofMap)
  ConstrainedTProductDofMap(similar(i.map),i.dof_to_constraint_mask)
end

get_univariate_dof_map(i::ConstrainedTProductDofMap) = get_univariate_dof_map(i.map)

# nnz bounding boxes

function find_free_values_range(inds,diri_entities,d)
  s_d = size(inds,d)
  entity_d = view(diri_entities,d,:)
  start = first(entity_d) ? 2 : 1
  finish = last(entity_d) ? s_d-1 : s_d
  start:finish
end

function find_free_values_box(
  inds::AbstractArray{Ti,D},
  diri_entities::AbstractMatrix{Bool}
  ) where {Ti,D}

  D′ = size(diri_entities,1)
  ranges = ntuple(d -> find_free_values_range(inds,diri_entities,d),D′)
  if D != D′
    @assert D == D′+1
    box = LinearIndices(inds)[ranges...,:]
  else
    box = LinearIndices(inds)[ranges...]
  end
  return box
end
