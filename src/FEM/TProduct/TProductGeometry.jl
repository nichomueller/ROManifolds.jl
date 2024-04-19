function split(desc::CartesianDescriptor{D}) where D
  @unpack origin,sizes,partition,map,isperiodic = desc
  isotropy = is_isotropic(sizes) && is_isotropic(partition) && is_isotropic(map) && is_isotropic(isperiodic)
  if isotropy
    Fill(CartesianDescriptor(origin[1],sizes[1],partition[1],identity,isperiodic[1]),D)
  else
    map(d->CartesianDescriptor(origin[d],sizes[d],partition[d],map,isperiodic[d]),1:D)
  end
end

struct TensorProductDescriptor{A,B} <: GridapType
  factors::A
  desc::B
end

function TensorProductDescriptor(
  domain::NTuple{D},
  partition::NTuple{D};
  map::Function=identity,
  isperiodic::NTuple=tfill(false,Val(D))) where D

  desc = CartesianDescriptor(domain,partition;map,isperiodic)
  factors = split(desc)
  TensorProductDescriptor(factors,desc)
end

struct KroneckerCoordinates{D,T,A} <: AbstractTensorProductPoints{D,T,D}
  coords::A
  KroneckerCoordinates{D,T}(coords::A) where {D,T,A} = new{D,T,A}(coords)
end

function KroneckerCoordinates(coords::NTuple{D,CartesianCoordinates{1,T}}) where {D,T}
  KroneckerCoordinates{D,T}(coords)
end

function KroneckerCoordinates(coords::AbstractVector{CartesianCoordinates{1,T}}) where T
  D = length(coords)
  KroneckerCoordinates{D,T}(coords)
end

Base.size(a::KroneckerCoordinates,d::Integer) = size(a.coords,d)
Base.size(a::KroneckerCoordinates{D}) where D = ntuple(d->size(a,d),Val(D))
Base.IndexStyle(::Type{<:KroneckerCoordinates}) = IndexCartesian()

function Base.getindex(a::KroneckerCoordinates{D},I::Vararg{Integer,D}) where {D,T}
  p = zero(Mutable(Point{D,T}))
  @inbounds for d in 1:D
    p[d] = getindex(a.coords[d],I[d])
  end
end

function split(a::CartesianGrid)

end

struct TensorProductGrid{D,A,B} <: Grid{D,D}
  factors::A
  grid::B
  function TensorProductGrid(factors::A,grid::B) where {D,A,B<:TensorProductGrid{D}}
    new{D,A,B}(factors,grid)
  end
end

function TensorProductGrid(args...;kwargs...)
  grid = CartesianGrid(args...;kwargs...)
  factors = split(grid)
  TensorProductGrid(factors,grid)
end

Geometry.OrientationStyle(::Type{<:TensorProductGrid}) = Oriented()

function Geometry.get_cartesian_descriptor(a::TensorProductGrid)
  factors = map(get_cartesian_descriptor,a.factors)
  desc = get_cartesian_descriptor(a.grid)
  TensorProductDescriptor(factors,desc)
end

function Geometry.get_node_coordinates(a::TensorProductGrid)
  KroneckerCoordinates(map(get_node_coordinates,a.factors))
end

Geometry.get_cell_type(a::TensorProductGrid) = get_cell_type(a.grid)

Geometry.get_cell_node_ids(a::TensorProductGrid) = get_cell_node_ids(a.grid)

function Geometry.get_cell_map(a::TensorProductGrid)
  TensorProductAffineMap(map(get_cell_map,a.factors))
end

function Geometry.get_reffes(a::TensorProductGrid{D}) where D
  p = Polytope(tfill(HEX_AXIS,Val{D}()))
  order = 1
  reffe = TensorProductRefFE(p,lagrangian,order)
  [reffe,]
end

struct TensorProductMap{D,T,L,A} <: AbstractArray{TensorProductAffineMap{D,T,L},D}
  factors::A
  function TensorProductMap{D,T}(factors::A) where {D,T,A}
    L = D*D
    new{D,T,L}(factors)
  end
end

function TensorProductMap(factors::NTuple{D,TensorProductMap{1,T}}) where {D,T}
  TensorProductMap{D,T}(factors)
end

function TensorProductMap(factors::AbstractVector{<:TensorProductMap{1,T}}) where T
  D = length(factors)
  TensorProductMap{D,T}(factors)
end

Base.size(a::TensorProductMap,d::Integer) = size(a.factors,d)
Base.size(a::TensorProductMap{D}) where D = ntuple(d->size(a,d),Val(D))
Base.IndexStyle(::Type{<:TensorProductMap}) = IndexCartesian()

function Base.getindex(a::TensorProductMap{D,T},I::Vararg{Integer,D}) where {D,T}
  TensorProductAffineMap(map(getindex,a.factors,I))
end

struct TensorProductAffineMap{D,T,L} <: TensorProductField
  factors::Vector{AffineMap{1,T,1}}
  function TensorProductAffineMap(factors::Vector{AffineMap{1,T,1}}) where T
    D = length(factors)
    L = D^2
    new{D,T,L}(factors)
  end
end

function Fields.affine_map(gradient::AbstractVector,origin::AbstractVector)
  TensorProductAffineMap(map(affine_map,gradient,origin))
end

function _tensor_product(f::TensorProductAffineMap{D}) where D
  gradient = TensorValue(ntuple(d->f.gradient[d]...,Val(D)))
  origin = Point(ntuple(d->f.origin[d]...,Val(D)))
  AffineMap(gradient,origin)
end

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_cache(f::TensorProductAffineMap,x::$T)
      amap = _tensor_product(f)
      cache = return_cache(amap,x)
      amap,cache
    end
    function Arrays.evaluate!(_cache,f::TensorProductAffineMap,x::$T)
      amap,cache = _cache
      evaluate!(cache,amap,x)
    end
  end
end

function Arrays.return_cache(
  f::TensorProductAffineMap{D},
  x::Union{KroneckerCoordinates,TensorProductNodes}
  ) where D

  index_map = get_index_map(x)
  factors = get_factors(f)
  c = return_cache(factors[1],x.nodes[1])
  r = Vector{typeof(get_array(c))}(undef,D)
  return index_map,r,c
end

function Arrays.evaluate!(
  cache,
  f::TensorProductAffineMap,
  x::Union{KroneckerCoordinates,TensorProductNodes})

  setsize!(cache,size(x))
  y = cache.array
  G = f.gradient
  y0 = f.origin
  for i in eachindex(x)
    xi = x[i]
    yi = xi⋅G + y0
    y[i] = yi
  end
  y
end
