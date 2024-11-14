get_ordered_dof_map(f::SingleFieldFESpace) = TrivialDofMap(num_free_dofs(f))
get_ordered_dof_map(f::MultiFieldFESpace) = @notimplemented

abstract type AbstractDofMap{D,Ti,Tc} <: AbstractArray{Ti,D} end

FESpaces.ConstraintStyle(::Type{<:AbstractDofMap}) = UnConstrained()

function remove_constrained_dofs(i::AbstractDofMap)
  remove_constrained_dofs(i,ConstraintStyle(i))
end

function remove_constrained_dofs(i::AbstractDofMap,::UnConstrained)
  i
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

abstract type AbstractTrivialDofMap <: AbstractDofMap{1,Int,Float64} end

Base.getindex(i::AbstractTrivialDofMap,j::Integer) = j
Base.setindex!(i::AbstractTrivialDofMap,v::Integer,j::Integer) = nothing
Base.copy(i::AbstractTrivialDofMap) = i
Base.similar(i::AbstractTrivialDofMap) = i

struct TrivialDofMap <: AbstractTrivialDofMap
  length::Int
end

TrivialDofMap(i::AbstractArray) = TrivialDofMap(length(i))
Base.size(i::TrivialDofMap) = (i.length,)

struct DofMap{D,Ti,Tc,A<:AbstractArray{Bool}} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  dof_to_cell::Table{Ti,Vector{Ti},Vector{Ti}}
  free_vals_box::Array{Int,D}
  cell_to_mask::A
  dof_type::Type{Tc}
end

function DofMap(
  indices::AbstractArray,
  dof_to_cell::Table,
  cell_to_mask::AbstractArray,
  dof_type::Type=Float64)

  free_vals_box = find_free_values_box(indices)
  DofMap(indices,dof_to_cell,free_vals_box,cell_to_mask,dof_type)
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
    i.cell_to_mask,
    i.dof_type)
end

function Base.similar(i::DofMap)
  DofMap(
    similar(i.indices),
    i.dof_to_cell,
    i.free_vals_box,
    i.cell_to_mask,
    i.dof_type)
end

function CellData.change_domain(i::DofMap,t::Triangulation)
  cell_to_mask = get_cell_to_mask(t)
  DofMap(
    i.indices,
    i.dof_to_cell,
    i.free_vals_box,
    cell_to_mask,
    i.dof_type)
end

# optimization

const EntireDofMap{D,Ti,Tc,A<:Fill{Bool}} = DofMap{D,Ti,Tc,A}

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

const MultiValueDofMap{D,Ti,Tc<:MultiValue,A} = DofMap{D,Ti,Tc,A}

function _to_scalar_values!(i::AbstractArray,D::Integer,d::Integer)
  i .= (i .- d) ./ D .+ 1
end

function get_component(i::MultiValueDofMap{D},d;multivalue::Bool=true) where D
  ncomps = num_components(i)
  indices = collect(selectdim(i.indices,D,d))
  dof_to_first_owner_cell = collect(selectdim(i.indices,D,d))
  free_vals_box = collect(selectdim(i.free_vals_box,D,d))
  dof_type = eltype(i.dof_type)

  if !multivalue
    for j in i.free_vals_box
      indj = indices[j]
      indices[j] = (indj - d) / ncomps + 1
    end
  end

  DofMap(
    indices,
    dof_to_first_owner_cell,
    free_vals_box,
    i.cell_to_mask,
    dof_type)
end

struct ConstrainedDofMap{D,Ti,Tc,A} <: AbstractDofMap{D,Ti,Tc}
  map::DofMap{D,Ti,Tc,A}
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

# sparse maps

abstract type SparseDofMapStyle end
abstract type FullDofMapIndexing <: SparseDofMapStyle end
abstract type SparseDofMapIndexing <: SparseDofMapStyle end

# trivial case

struct TrivialSparseDofMap{A<:SparsityPattern} <: AbstractTrivialDofMap
  sparsity::A
end

TrivialDofMap(sparsity::SparsityPattern) = TrivialSparseDofMap(sparsity)
TrivialDofMap(i::TrivialSparseDofMap) = i
Base.size(i::TrivialSparseDofMap) = (nnz(i.sparsity),)

recast(a::AbstractArray,i::TrivialSparseDofMap) = recast(a,i.sparsity)

SparseDofMapStyle(i::TrivialSparseDofMap) = FullDofMapIndexing()

# non trivial case

"""
    SparseDofMap{D,Ti,A<:AbstractDofMap{D,Ti},B<:TProductSparsityPattern} <: AbstractDofMap{D,Ti}

Index map used to select the nonzero entries of a sparse matrix. The field `sparsity`
contains the tensor product sparsity of the matrix to be indexed. The field `indices`
refers to the nonzero entries of the sparse matrix, whereas `indices_sparse` is
used to access the corresponding sparse entries

"""
struct SparseDofMap{D,Ti,A<:SparsityPattern,B<:SparseDofMapStyle} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  indices_sparse::Array{Ti,D}
  sparsity::A
  index_style::B
end

function SparseDofMap(
  indices::AbstractArray,
  indices_sparse::AbstractArray,
  sparsity::SparsityPattern)

  index_style = FullDofMapIndexing()
  SparseDofMap(indices,indices_sparse,sparsity,index_style)
end

# reindexing

SparseDofMapStyle(i::SparseDofMap) = i.index_style

function FullDofMap(i::SparseDofMap)
  SparseDofMap(i.indices,i.indices_sparse,i.sparsity,FullDofMapIndexing())
end

function SparseDofMap(i::SparseDofMap)
  SparseDofMap(i.indices,i.indices_sparse,i.sparsity,SparseDofMapIndexing())
end

Base.size(i::SparseDofMap) = size(i.indices)

function Base.getindex(i::SparseDofMap{D,Ti,A,FullDofMap},j::Vararg{Integer,D}) where {D,Ti,A}
  getindex(i.indices,j...)
end

function Base.setindex!(i::SparseDofMap{D,Ti,A,FullDofMap},v,j::Vararg{Integer,D}) where {D,Ti,A}
  setindex!(i.indices,v,j...)
end

function Base.getindex(i::SparseDofMap{D,Ti,A,SparseDofMap},j::Vararg{Integer,D}) where {D,Ti,A}
  getindex(i.indices_sparse,j...)
end

function Base.setindex!(i::SparseDofMap{D,Ti,A,SparseDofMap},v,j::Vararg{Integer,D}) where {D,Ti,A}
  setindex!(i.indices_sparse,v,j...)
end

function Base.copy(i::SparseDofMap)
  SparseDofMap(copy(i.indices),copy(i.indices_sparse),i.sparsity,i.index_style)
end

function Base.similar(i::SparseDofMap)
  SparseDofMap(similar(i.indices),similar(i.indices_sparse),i.sparsity,i.index_style)
end

recast(A::AbstractArray,i::SparseDofMap) = recast(A,i.sparsity)

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
