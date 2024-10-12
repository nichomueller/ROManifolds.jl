"""
    abstract type Contribution end

Represents quantitites whose values vary upon a triangulation. The values can
be accessed by indexing a the corresponding triangulation. See [`DomainContribution`](@ref)
in [`Gridap`](@ref) for more details. Subtypes:
- [`ArrayContribution`](@ref)
- [`AffineContribution`](@ref)

"""
abstract type Contribution end

CellData.get_domains(a::Contribution) = a.trians
get_values(a::Contribution) = a.values

Base.length(a::Contribution) = length(a.values)
Base.size(a::Contribution,i...) = size(a.values,i...)
Base.getindex(a::Contribution,i...) = a.values[i...]
Base.setindex!(a::Contribution,v,i...) = a.values[i...] = v
Base.eachindex(a::Contribution) = eachindex(a.values)

# Allow do-block syntax
@inline function contribution(f,trians)
  values = map(f,trians)
  Contribution(values,trians)
end

function contribution!(a,values)
  a.values .= values
end

function contribution!(a,f,trians)
  contribution!(a,map(f,trians))
end

function Base.getindex(a::Contribution,trian::Triangulation...)
  perm = find_trian_permutation(trian,a.trians)
  getindex(a,perm...)
end

Contribution(v::V,t::Triangulation) where V = Contribution((v,),(t,))

function Contribution(
  v::Tuple{Vararg{AbstractArray{T,N}}},
  t::Tuple{Vararg{Triangulation}}) where {T,N}

  ArrayContribution{T,N}(v,t)
end

function Contribution(
  v::Tuple{Vararg{ArrayBlock{T,N}}},
  t::Tuple{Vararg{Triangulation}}) where {T,N}

  ArrayContribution{T,N}(v,t)
end

"""
    struct ArrayContribution{T,N,V,K} <: Contribution end

Contribution whose values are arrays and/or parametric arrays.

"""
struct ArrayContribution{T,N,V,K} <: Contribution
  values::V
  trians::K
  function ArrayContribution{T,N}(values::V,trians::K) where {T,N,V,K}
    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{T,N,V,K}(values,trians)
  end
end

const VectorContribution{T,V,K} = ArrayContribution{T,1,V,K}
const MatrixContribution{T,V,K} = ArrayContribution{T,2,V,K}

Base.eltype(::ArrayContribution{T}) where T = T
Base.eltype(::Type{<:ArrayContribution{T}}) where T = T
Base.ndims(::ArrayContribution{T,N}) where {T,N} = N
Base.ndims(::Type{<:ArrayContribution{T,N}}) where {T,N} = N
Base.copy(a::ArrayContribution) = Contribution(copy(a.values),a.trians)

Base.sum(a::ArrayContribution) = sum(a.values)

function Base.fill!(a::ArrayContribution,v)
  for vals in a.values
    fill!(vals,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::ArrayContribution,v)
  for vals in a.values
    LinearAlgebra.fillstored!(vals,v)
  end
  a
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::AbstractVector,
  α::Number,β::Number)

  for c in c.values, a in a.values
    mul!(c,a,b,α,β)
  end
  a
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::VectorContribution,
  α::Number,β::Number)

  @check length(c) == length(b)
  for (c,b) in zip(c.values,b.values), a in a.values
    mul!(c,a,b,α,β)
  end
  a
end

function LinearAlgebra.axpy!(α::Number,a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (a,b) in (a.values,b.values)
    axpy!(α,a,b)
  end
  b
end

function Algebra.copy_entries!(a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (a,b) in zip(a.values,b.values)
    copy_entries!(a,b)
  end
  a
end

struct ContributionBroadcast{D,T}
  contrib::D
  trians::T
end

function Base.broadcasted(f,a::ArrayContribution,b::Number)
  ContributionBroadcast(map(values -> Base.broadcasted(f,values,b),a.values),a.trians)
end

function Base.materialize(c::ContributionBroadcast)
  Contribution(map(Base.materialize,c.contrib),c.trians)
end

function Base.materialize!(a::ArrayContribution,c::ContributionBroadcast)
  @check a.trians === c.trians
  map(Base.materialize!,a.values,c.contrib)
  a
end

# small hack, it allows to efficiently deal with jacobians in an unsteady setting
const TupOfArrayContribution{T} = Tuple{Vararg{ArrayContribution{T}}}

Base.eltype(::TupOfArrayContribution{T}) where T = T
Base.eltype(::Type{<:TupOfArrayContribution{T}}) where T = T

function LinearAlgebra.fillstored!(a::TupOfArrayContribution,v)
  for ai in a
    LinearAlgebra.fillstored!(ai,v)
  end
  a
end

function Algebra.copy_entries!(a::TupOfArrayContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  for (a,b) in zip(a,b)
    copy_entries!(a,b)
  end
  a
end

# triangulation utils

"""
    is_parent(tparent::Triangulation,tchild::Triangulation) -> Bool

Returns true if `tchild` is a triangulation view of `tparent`, false otherwise

"""
function is_parent(tparent::Triangulation,tchild::Triangulation)
  false
end

function is_parent(
  tparent::BodyFittedTriangulation,
  tchild::BodyFittedTriangulation{Dt,Dp,A,<:Geometry.GridView}) where {Dt,Dp,A}
  tparent.grid === tchild.grid.parent
end

function is_parent(tparent::BoundaryTriangulation,tchild::Geometry.TriangulationView)
  tparent === tchild.parent
end

function get_parent(t::Geometry.Grid)
  @abstractmethod
end

function get_parent(gv::Geometry.GridView)
  gv.parent
end

function get_parent(t::Geometry.TriangulationView)
  t.parent
end

function get_parent(t::BodyFittedTriangulation)
  grid = get_parent(get_grid(t))
  model = get_background_model(t)
  tface_to_mface = IdentityVector(num_cells(grid))
  BodyFittedTriangulation(model,grid,tface_to_mface)
end

function get_parent(t::AbstractVector{<:Triangulation})
  get_parent(first(t))
end

"""
    is_parent(t::Triangulation) -> Triangulation

When `t` is a triangulation view, returns its parent; throws an error when `t`
is not a triangulation view

"""
function get_parent(t::Triangulation)
  @abstractmethod
end

function Base.isapprox(t::Grid,s::Grid)
  false
end

function Base.isapprox(t::T,s::T) where {T<:UnstructuredGrid}
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_types == s.cell_types &&
    t.node_coordinates == s.node_coordinates
  )
end

function Base.isapprox(t::T,s::T) where {T<:Geometry.GridView}
  t.parent ≈ s.parent
end

get_tface_to_mface(t::Triangulation) = @abstractmethod "$(typeof(t))"
get_tface_to_mface(t::BodyFittedTriangulation) = t.tface_to_mface
get_tface_to_mface(t::BoundaryTriangulation) = get_tface_to_mface(t.trian)

function Base.isapprox(t::T,s::S) where {T<:Triangulation,S<:Triangulation}
  false
end

function Base.isapprox(t::T,s::T) where {T<:Triangulation}
  get_tface_to_mface(t) == get_tface_to_mface(s) && get_grid(t) ≈ get_grid(s)
end

function Base.isapprox(t::T,s::T) where {T<:Geometry.TriangulationView}
  get_view_indices(t) == get_view_indices(s) && get_parent(t) ≈ get_parent(s)
end

function Base.isapprox(t::T,s::T) where {T<:BoundaryTriangulation}
  (
    t.trian.tface_to_mface == s.trian.tface_to_mface &&
    t.trian.grid.parent.cell_node_ids == s.trian.grid.parent.cell_node_ids &&
    t.trian.grid.parent.cell_types == s.trian.grid.parent.cell_types &&
    t.trian.grid.parent.node_coordinates == s.trian.grid.parent.node_coordinates
  )
end

function isapprox_parent(tparent::Triangulation,tchild::Triangulation)
  tparent ≈ get_parent(tchild)
end

function get_view_indices(t::BodyFittedTriangulation)
  grid = get_grid(t)
  grid.cell_to_parent_cell
end

function get_view_indices(t::Geometry.TriangulationView)
  t.cell_to_parent_cell
end

function get_union_indices(trians)
  indices = map(get_view_indices,trians)
  union(indices...) |> unique
end

"""
    merge_triangulations(trians::Tuple{Vararg{Triangulation}}) -> Triangulation

Given a tuple of triangulation views `trians`, returns the triangulation view
on the union of the viewed cells. In other words, the minimum common integration
domain is found

"""
function merge_triangulations(trians)
  parent = get_parent(trians)
  uindices = get_union_indices(trians)
  view(parent,uindices)
end

function find_trian_permutation(a,b,cmp::Function)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

function find_trian_permutation(a,b)
  cmp = (a,b) -> a == b || is_parent(a,b)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

"""
    order_triangulations(tparents::Tuple{Vararg{Triangulation}},
      tchildren::Tuple{Vararg{Triangulation}}) -> Tuple{Vararg{Triangulation}}

Orders the triangulation children in the same way as the triangulation parents

"""
function order_triangulations(tparents,tchildren)
  @check length(tparents) == length(tchildren)
  iperm = find_trian_permutation(tparents,tchildren)
  map(iperm->tchildren[iperm],iperm)
end

"""
    find_closest_view(tparents::Tuple{Vararg{Triangulation}},
      tchild::Triangulation) -> Integer, Triangulation

Finds the approximate parent of `tchild`; it returns the parent's index and its
view in the same indices as `tchild` (which should be a triangulation view)

"""
function find_closest_view(tparents,tchild::Triangulation)
  cmp(a,b) = isapprox_parent(b,a)
  iperm::Tuple{Vararg{Int}} = find_trian_permutation((tchild,),tparents,cmp)
  @check length(iperm) == 1
  indices = get_view_indices(tchild)
  return iperm,view(tparents[iperm...],indices)
end
