abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

FESpaces.ConstraintStyle(i::I) where I<:AbstractDofMap = ConstraintStyle(I)
FESpaces.ConstraintStyle(::Type{<:AbstractDofMap}) = UnConstrained()

function remove_constrained_dofs(i::AbstractDofMap)
  remove_constrained_dofs(i,ConstraintStyle(i))
end

function remove_constrained_dofs(i::AbstractDofMap,::UnConstrained)
  collect(i)
end

function remove_constrained_dofs(i::AbstractDofMap,::Constrained)
  @abstractmethod
end

dofs_to_constrained_dofs(i::AbstractDofMap) = @abstractmethod

TensorValues.num_components(::Type{<:AbstractDofMap{D,Ti}}) where {D,Ti} = num_components(Ti)

function vectorize(i::AbstractDofMap)
  vec(remove_constrained_dofs(i))
end

function invert(i::AbstractDofMap)
  invert(i,ConstraintStyle(i))
end

function invert(i::AbstractDofMap,::UnConstrained)
  i′ = similar(i)
  s′ = size(i)
  invi = invperm(vectorize(i))
  copyto!(i′,reshape(invi,s′))
  i′
end

function invert(i::AbstractDofMap,::Constrained)
  i′ = similar(i)
  s′ = size(i)
  dof_to_mask = dofs_to_constrained_dofs(i)
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

# trivial map

abstract type AbstractTrivialDofMap <: AbstractDofMap{1,Int} end

Base.getindex(i::AbstractTrivialDofMap,j::Integer) = j
Base.setindex!(i::AbstractTrivialDofMap,v::Integer,j::Integer) = nothing
Base.copy(i::AbstractTrivialDofMap) = i
Base.similar(i::AbstractTrivialDofMap) = i

struct TrivialDofMap <: AbstractTrivialDofMap
  length::Int
end

TrivialDofMap(i::AbstractArray) = TrivialDofMap(length(i))
Base.size(i::TrivialDofMap) = (i.length,)

struct DofMap{D,Ti,A<:AbstractArray{Bool}} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  free_vals_box::Array{Int,D}
  cell_to_mask::A
end

function DofMap(
  indices::AbstractArray,
  dof_to_cell::Table,
  cell_to_mask::AbstractArray)

  free_vals_box = find_free_values_box(indices)
  DofMap(indices,dof_to_cell,free_vals_box,cell_to_mask)
end

Base.IndexStyle(i::DofMap) = IndexLinear()
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

@inline function show_dof(i::DofMap,idof::Integer)
  pini = i.dof_to_cell.ptrs[idof]
  pend = i.dof_to_cell.ptrs[idof+1]-1

  show = false
  for j in pini:pend
    if !i.cell_to_mask[i.dof_to_cell.data[j]]
      show = true
      break
    end
  end

  return show
end

function Base.copy(i::DofMap)
  DofMap(
    copy(i.indices),
    i.dof_to_cell,
    i.free_vals_box,
    i.cell_to_mask)
end

function Base.similar(i::DofMap)
  DofMap(
    similar(i.indices),
    i.dof_to_cell,
    i.free_vals_box,
    i.cell_to_mask)
end

function CellData.change_domain(i::DofMap,t::Triangulation)
  cell_to_mask = get_cell_to_mask(t)
  DofMap(
    i.indices,
    i.dof_to_cell,
    i.free_vals_box,
    cell_to_mask)
end

# optimization

const EntireDofMap{D,Ti,A<:Fill{Bool}} = DofMap{D,Ti,A}

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

function get_component(i::DofMap{D};to_scalar=false) where D
  ncomps = size(i,D)
  indices = collect(selectdim(i.indices,D,ncomps))
  free_vals_box = collect(selectdim(i.free_vals_box,D,1))

  if to_scalar
    for j in free_vals_box
      indj = indices[j]
      indices[j] = indj / ncomps
    end
  end

  DofMap(
    indices,
    i.dof_to_cell,
    free_vals_box,
    i.cell_to_mask)
end

struct ConstrainedDofMap{D,Ti,A} <: AbstractDofMap{D,Ti}
  map::DofMap{D,Ti,A}
  dof_to_constraint_mask::Vector{Bool}
end

FESpaces.ConstraintStyle(::Type{<:ConstrainedDofMap}) = Constrained()

get_dof_to_constraints(i::ConstrainedDofMap) = i.dof_to_constraint_mask

Base.size(i::ConstrainedDofMap) = size(i.map)

function CellData.change_domain(i::ConstrainedDofMap,t::Triangulation)
  ConstrainedDofMap(change_domain(i.map,t),i.dof_to_constraint_mask)
end

function Base.getindex(i::ConstrainedDofMap{D,Ti},j::Vararg{Integer,D}) where {D,Ti}
  getindex(i.map,j...)
end

function Base.setindex!(i::ConstrainedDofMap{D,Ti},v,j::Vararg{Integer,D}) where {D,Ti}
  !(i.dof_to_constraint_mask[j...]) && setindex!(i.map,v,j...)
end

function Base.copy(i::ConstrainedDofMap)
  ConstrainedDofMap(copy(i.map),i.dof_to_constraint_mask)
end

function Base.similar(i::ConstrainedDofMap)
  ConstrainedDofMap(similar(i.map),i.dof_to_constraint_mask)
end

# tensor product index map, a lot of previous machinery is not needed

struct TProductDofMap{D,Ti,I<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  indices_1d::Vector{Vector{Ti}}
end

Base.size(i::TProductDofMap) = size(i.indices)

function Base.getindex(i::TProductDofMap{D,Ti},j::Vararg{Integer,D}) where {D,Ti}
  getindex(i.indices,j...)
end

function Base.setindex!(i::TProductDofMap{D,Ti},v,j::Vararg{Integer,D}) where {D,Ti}
  setindex!(i.indices,v,j...)
end

function Base.copy(i::TProductDofMap)
  TProductDofMap(copy(i.indices),i.indices_1d,i.dof_map)
end

function Base.similar(i::TProductDofMap)
  TProductDofMap(similar(i.indices),i.indices_1d,i.dof_map)
end

# nnz bounding boxes

function get_extrema(i::AbstractArray,d::Int)
  dslices = eachslice(i,dims=d)
  start = dslices[1]
  finish = dslices[end]
  return (start,finish)
end

isneg(a::Number) = a ≤ 0

function find_free_values_range(inds::AbstractArray,d::Int)
  istart,ifinish = get_extrema(inds,d)
  allneg_left = true
  allneg_right = true
  i = 1
  while i ≤ length(istart) && (allneg_left || allneg_right)
    if allneg_left
      ileft = @views istart[i]
      allneg_left = isneg(ileft) ? allneg_left : !allneg_left
    end
    if allneg_right
      iright = @views ifinish[i]
      allneg_right = isneg(iright) ? allneg_right : !allneg_right
    end
    i += 1
  end
  start = allneg_left ? 2 : 1
  finish = allneg_right ? size(inds,d)-1 : size(inds,d)
  return start:finish
end

function find_free_values_box(inds::AbstractArray{Ti,D}) where {Ti,D}
  ranges = ntuple(d -> find_free_values_range(inds,d),D)
  box = LinearIndices(inds)[ranges...]
  return box
end

function find_free_values_box(inds::AbstractVector{Ti}) where {Ti}
  ranges = find_free_values_range(inds,1)
  box = collect(ranges)
  return box
end
