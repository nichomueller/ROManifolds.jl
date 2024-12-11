abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

Base.IndexStyle(i::AbstractDofMap) = IndexLinear()

FESpaces.ConstraintStyle(i::I) where I<:AbstractDofMap = ConstraintStyle(I)
FESpaces.ConstraintStyle(::Type{<:AbstractDofMap}) = UnConstrained()

function remove_constrained_dofs(i::AbstractDofMap)
  remove_constrained_dofs(i,ConstraintStyle(i))
end

function remove_constrained_dofs(i::AbstractDofMap,::UnConstrained)
  collect(i)
end

function remove_constrained_dofs(i::AbstractDofMap{Ti},::Constrained) where Ti
  dof_to_constraint = get_dof_to_constraints(i)
  nconstraints = findall(dof_to_constraint)
  i′ = zeros(Ti,length(i)-nconstraints)
  count = 0
  for j in i
    if !dof_to_constraint[j]
      count += 1
      i′[count] = j
    end
  end
  i′
end

get_dof_to_constraints(i::AbstractDofMap) = @abstractmethod

TensorValues.num_components(::Type{<:AbstractDofMap{D,Ti}}) where {D,Ti} = num_components(Ti)

function vectorize(i::AbstractDofMap)
  vec(remove_constrained_dofs(i))
end

function invert(i::AbstractDofMap)
  invert(i,ConstraintStyle(i))
end

function invert(i::AbstractDofMap,::UnConstrained)
  s = size(i)
  i′ = zeros(eltype(i),s)
  invi = invperm(vectorize(i))
  copyto!(i′,reshape(invi,s))
  i′
end

function invert(i::AbstractDofMap,::Constrained)
  i′ = similar(i)
  s′ = size(i)
  dof_to_mask = dof_to_constraint_mask(i)
  invi = invperm(vectorize(i))
  nconstrained = 0
  for k in eachindex(i)
    if dof_to_mask[k]
      i′[k] = i[k]
      nconstrained += 1
    else
      i′[k] = invi[k-nconstrained]
    end
  end
  i′
end

get_dof_to_cell(i::AbstractDofMap) = @abstractmethod
get_tface_to_mask(i::AbstractDofMap) = @abstractmethod
Utils.get_tface_to_mface(i::AbstractDofMap) = @abstractmethod

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

# trivial map

abstract type AbstractTrivialDofMap{Ti} <: AbstractDofMap{1,Ti} end

Base.setindex!(i::AbstractTrivialDofMap,v,j::Integer) = v
Base.copy(i::AbstractTrivialDofMap) = i
Base.similar(i::AbstractTrivialDofMap) = i

struct TrivialDofMap{Ti,A<:AbstractVector{Bool}} <: AbstractTrivialDofMap{Ti}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  tface_to_mask::Vector{Bool}
  tface_to_mface::A
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

struct DofMap{D,Ti,A<:AbstractVector} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  free_vals_box::Array{Int,D}
  tface_to_mask::Vector{Bool}
  tface_to_mface::A
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

# optimization

const EntireDofMap{D,Ti,A<:IdentityVector} = DofMap{D,Ti,A}

Base.@propagate_inbounds function Base.getindex(
  i::EntireDofMap{D,Ti},
  j::Vararg{Integer,D}
  ) where {D,Ti}

  free_j = i.free_vals_box[j...]
  i.indices[free_j]
end

Base.@propagate_inbounds function Base.setindex!(
  i::EntireDofMap{D,Ti},
  v,j::Vararg{Integer,D}
  ) where {D,Ti}

  free_j = i.free_vals_box[j...]
  i.indices[free_j] = v
  v
end

# multi value

function _to_scalar_values!(i::AbstractArray,D::Integer,d::Integer)
  i .= (i .- d) ./ D .+ 1
end

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

struct ConstrainedDofMap{D,Ti,A} <: AbstractDofMap{D,Ti}
  map::DofMap{D,Ti,A}
  dof_to_constraint_mask::Vector{Bool}
end

FESpaces.ConstraintStyle(::Type{<:ConstrainedDofMap}) = Constrained()

get_dof_to_constraints(i::ConstrainedDofMap) = i.dof_to_constraint_mask

Base.size(i::ConstrainedDofMap) = size(i.map)

function Base.getindex(i::ConstrainedDofMap,j::Integer)
  getindex(i.map,j...)
end

function Base.setindex!(i::ConstrainedDofMap,v,j::Integer)
  setindex!(i.map,v,j)
end

function Base.copy(i::ConstrainedDofMap)
  ConstrainedDofMap(copy(i.map),i.dof_to_constraint_mask)
end

function Base.similar(i::ConstrainedDofMap)
  ConstrainedDofMap(similar(i.map),i.dof_to_constraint_mask)
end

function CellData.change_domain(i::ConstrainedDofMap,t::Triangulation)
  change_domain(i.map,t)
end

# tensor product index map, a lot of previous machinery is not needed

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

# nnz bounding boxes

function find_free_values_range(inds,diri_entities,d)
  s_d = size(inds,d)
  entity_d = view(diri_entities,:,d)
  start = first(entity_d) ? 2 : 1
  finish = last(entity_d) ? s_d-1 : s_d
  start:finish
end

function find_free_values_box(
  inds::AbstractArray{Ti,D},
  diri_entities::AbstractMatrix{Bool}
  ) where {Ti,D}

  D′ = size(diri_entities,2)
  ranges = ntuple(d -> find_free_values_range(inds,diri_entities,d),D′)
  if D != D′
    @assert D == D′+1
    box = LinearIndices(inds)[ranges...,:]
  else
    box = LinearIndices(inds)[ranges...]
  end
  return box
end
