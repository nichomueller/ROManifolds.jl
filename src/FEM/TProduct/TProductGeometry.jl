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
  isotropy = Isotropic()
  TensorProductDescriptor(factors,desc,isotropy)
end

function Geometry.get_node_coordinates(a::TensorProductGrid)
  KroneckerCoordinates(map(get_node_coordinates,a.factors))
end

Geometry.get_cell_type(a::TensorProductGrid) = get_cell_type(a.grid)

Geometry.get_cell_node_ids(a::TensorProductGrid) = get_cell_node_ids(a.grid)

Geometry.get_cell_map(a::TensorProductGrid) = TensorProductCellMap(map(get_cell_map,a.factors))

function Geometry.get_reffes(a::TensorProductGrid{D}) where D
  p = Polytope(tfill(HEX_AXIS,Val(D)))
  reffe = TensorProductRefFE(p,lagrangian,Float64,1)
  [reffe,]
end

struct TensorProductCellMap{D,T,A} <:AbstractVector{Geometry.CartesianMap{1,T,1}}
  factors::A
  function TensorProductCellMap(factors::A) where {T,A<:AbstractVector{Geometry.CartesianMap{1,T,1}}}
    D = length(factors)
    new{D,T,A}(factors)
  end
end

get_factors(a::TensorProductCellMap) = a.factors
Base.size(a::TensorProductCellMap) = (length(a.factors),)
Base.getindex(a::TensorProductCellMap,i::Int) = getindex(get_factors(a),i)

function Arrays.lazy_map(::typeof(∇),a::TensorProductCellMap{D}) where D
  factors = map(lazy_map,Fill(∇,D),get_factors(a))
  GenericTensorProductField(factors)
end

function Arrays.lazy_map(::typeof(∇),a::LazyArray{<:Fill{<:Reindex{<:TensorProductCellMap{D}}}}) where D
  i_to_map = a.maps.value.values
  j_to_i = a.args[1]
  i_to_grad = lazy_map(∇,i_to_map)
  factors = map(lazy_map,Fill(Reindex(i_to_grad),D),j_to_i)
  GenericTensorProductField(factors)
end

struct TensorProductDiscreteModel{D,A,B} <: DiscreteModel{D,D}
  model::A
  grid::B
  function TensorProductDiscreteModel(model::A,grid::B
    ) where {D,A<:CartesianDiscreteModel{D},B<:TensorProductGrid{D}}
    new{D,A,B}(model,grid)
  end
end

function TensorProductDiscreteModel(args...;kwargs...)
  model = CartesianDiscreteModel(args...;kwargs...)
  grid = TensorProductGrid(args...;kwargs...)
  TensorProductDiscreteModel(model,grid)
end

Geometry.get_grid(model::TensorProductDiscreteModel) = model.grid
Geometry.get_grid_topology(model::TensorProductDiscreteModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::TensorProductDiscreteModel) = get_face_labeling(model.model)
