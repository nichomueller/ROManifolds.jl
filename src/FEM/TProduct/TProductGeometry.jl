function split(desc::CartesianDescriptor{D}) where D
  @unpack origin,sizes,partition,map,isperiodic = desc
  _split_cartesian_descriptor(origin,sizes,partition,map,isperiodic)
end

struct TensorProductDescriptor{I,A,B} <: GridapType
  factors::A
  desc::B
  isotropy::I
end

get_factors(a::TensorProductDescriptor) = a.factors
Geometry.get_cartesian_descriptor(a::TensorProductDescriptor) = a.desc

function TensorProductDescriptor(
  domain,
  partition::NTuple{D};
  map::Function=identity,
  isperiodic::NTuple=tfill(false,Val(D))) where D

  desc = CartesianDescriptor(domain,partition;map,isperiodic)
  factors,isotropy = split(desc)
  TensorProductDescriptor(factors,desc,isotropy)
end

struct KroneckerCoordinates{D,I,T,A} <: AbstractTensorProductPoints{D,I,T,D}
  coords::A
  isotropy::I
  function KroneckerCoordinates{D,T}(coords::A,isotropy::I=Isotropy(coords)) where {D,I,T,A}
    new{D,I,T,A}(coords,isotropy)
  end
end

function KroneckerCoordinates(coords::NTuple{D,Geometry.CartesianCoordinates{1,T}},args...) where {D,T}
  KroneckerCoordinates{D,T}(coords,args...)
end

function KroneckerCoordinates(coords::AbstractVector{<:Geometry.CartesianCoordinates{1,T}},args...) where T
  D = length(coords)
  KroneckerCoordinates{D,T}(coords,args...)
end

get_factors(a::KroneckerCoordinates) = a.factors

function get_indices_map(a::KroneckerCoordinates{D}) where D
  polytope = Polytope(tfill(HEX_AXIS,Val(D)))
  orders = map(length,get_factors(a)) .- 1
  trivial_nodes_map(;polytope,orders)
end

function tensor_product_points(::Type{<:KroneckerCoordinates},coords,::NodesMap)
  KroneckerCoordinates(coords)
end

Base.size(a::KroneckerCoordinates,d::Integer) = size(a.coords[d],1)
Base.size(a::KroneckerCoordinates{D}) where D = ntuple(d->size(a,d),Val(D))
Base.IndexStyle(::Type{<:KroneckerCoordinates}) = IndexCartesian()

function Base.getindex(a::KroneckerCoordinates{D,I,T},i::Vararg{Integer,D}) where {D,I,T}
  p = zero(Mutable(Point{D,T}))
  @inbounds for d in 1:D
    pd = getindex(a.coords[d],i[d])
    p[d] = pd.data[1]
  end
  return Point(p)
end

struct TensorProductGrid{D,A,B} <: Grid{D,D}
  factors::A
  grid::B
  function TensorProductGrid(factors::A,grid::B) where {D,A,B<:CartesianGrid{D}}
    new{D,A,B}(factors,grid)
  end
end

function TensorProductGrid(args...;kwargs...)
  desc = TensorProductDescriptor(args...;kwargs...)
  grid = CartesianGrid(get_cartesian_descriptor(desc))
  factors = map(CartesianGrid,get_factors(desc))
  TensorProductGrid(factors,grid)
end

get_factors(a::TensorProductGrid) = a.factors
Geometry.get_grid(a::TensorProductGrid) = a.grid
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

Geometry.get_cell_map(a::TensorProductGrid) = get_cell_map(a.grid)

function Geometry.get_reffes(a::TensorProductGrid{D}) where D
  p = Polytope(tfill(HEX_AXIS,Val{D}()))
  reffe = TensorProductRefFE(p,lagrangian,Float64,1)
  [reffe,]
end
