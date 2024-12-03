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
get_cell_to_mask(i::AbstractDofMap) = @abstractmethod

@inline function show_dof(i::AbstractDofMap,idof::Integer)
  dof_to_cell = get_dof_to_cell(i)
  cell_to_mask = get_cell_to_mask(i)

  pini = dof_to_cell.ptrs[idof]
  pend = dof_to_cell.ptrs[idof+1]-1

  show = false
  for j in pini:pend
    if !cell_to_mask[dof_to_cell.data[j]]
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

struct TrivialDofMap{Ti,A<:AbstractArray{Bool}} <: AbstractTrivialDofMap{Ti}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  cell_to_mask::A
end

function TrivialDofMap(i::AbstractDofMap)
  TrivialDofMap(get_dof_to_cell(i),get_cell_to_mask(i))
end

Base.size(i::TrivialDofMap) = (length(i.dof_to_cell),)

Base.@propagate_inbounds function Base.getindex(i::TrivialDofMap{Ti},j::Integer) where Ti
  show_dof(i,j) ? j : zero(Ti)
end

get_dof_to_cell(i::TrivialDofMap) = i.dof_to_cell
get_cell_to_mask(i::TrivialDofMap) = i.cell_to_mask

function CellData.change_domain(i::TrivialDofMap,t::Triangulation)
  cell_to_mask = get_cell_to_mask(t)
  TrivialDofMap(i.dof_to_cell,cell_to_mask)
end

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
  DofMap(copy(i.indices),i.dof_to_cell,i.free_vals_box,i.cell_to_mask)
end

function Base.similar(i::DofMap)
  DofMap(similar(i.indices),i.dof_to_cell,i.free_vals_box,i.cell_to_mask)
end

function CellData.change_domain(i::DofMap,t::Triangulation)
  cell_to_mask = get_cell_to_mask(t)
  DofMap(i.indices,i.dof_to_cell,i.free_vals_box,cell_to_mask)
end

get_dof_to_cell(i::DofMap) = i.dof_to_cell
get_cell_to_mask(i::DofMap) = i.cell_to_mask

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

  DofMap(indices,dof_to_cell,free_vals_box,i.cell_to_mask)
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
