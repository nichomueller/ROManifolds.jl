struct UnivariateDescriptor{D,T,F<:Function} <: AbstractVector{CartesianDescriptor{1,T,F}}
  origin::Point{D,T}
  sizes::NTuple{D,T}
  partition::NTuple{D,Int}
  map::F
  isperiodic::NTuple{D,Bool}

  function UnivariateDescriptor(
    origin::Point{D},
    sizes::NTuple{D},
    partition;
    map::Function=identity,
    isperiodic::NTuple{D,Bool}=tfill(false,Val{D}())) where D

    if map != identity || any(isperiodic)
      @notimplemented "This case is not yet implemented"
    end

    T = eltype(sizes)
    F = typeof(map)
    new{D,T,F}(origin,sizes,Tuple(partition),map,isperiodic)
  end
end

function UnivariateDescriptor(
  domain,
  partition;
  map::Function=identity,
  isperiodic::NTuple=tfill(false,Val{length(partition)}()))

  D = length(partition)
  limits = [(domain[2*d-1],domain[2*d]) for d in 1:D]
  sizes = Tuple([(limits[d][2]-limits[d][1])/partition[d] for d in 1:D])
  origin = Point([ limits[d][1] for d in 1:D]...)
  UnivariateDescriptor(origin,sizes,partition;map=map,isperiodic=isperiodic)
end

function UnivariateDescriptor(
  pmin::Point{D},
  pmax::Point{D},
  partition;
  kwargs...) where D

  T = eltype(pmin)
  domain = zeros(T,2*D)
  for d in 1:D
    domain[2*(d-1)+1] = pmin[d]
    domain[2*(d-1)+2] = pmax[d]
  end
  UnivariateDescriptor(domain,partition;kwargs...)
end

Base.length(k::UnivariateDescriptor{D}) where D = D
Base.size(k::UnivariateDescriptor{D}) where D = (D,)
Base.IndexStyle(::Type{<:UnivariateDescriptor}) = IndexLinear()

function Base.getindex(k::UnivariateDescriptor,d::Integer)
  origin = Point(k.origin[d])
  size = Tuple(k.sizes[d])
  partition = k.partition[d]
  map = k.map
  isperiodic = Tuple(k.isperiodic[d])
  CartesianDescriptor(origin,size,partition;map,isperiodic)
end

struct UnivariateCoordinates{D,T,F} <: AbstractVector{Vector{Point{1,T}}}
  data::UnivariateDescriptor{D,T,F}
end

Base.length(a::UnivariateCoordinates{D}) where D = D
Base.size(a::UnivariateCoordinates{D}) where D = (D,)
Base.IndexStyle(::Type{<:UnivariateCoordinates}) = IndexLinear()

function Base.getindex(a::UnivariateCoordinates,d::Integer)
  desc = a.data[d]
  Geometry.CartesianCoordinates(desc)
end

struct UnivariateCellNodes{D} <: AbstractVector{Vector{Vector{Int32}}}
  partition::NTuple{D,Int}
  function UnivariateCellNodes(partition)
    D = length(partition)
    new{D}(partition)
  end
end

Base.length(a::UnivariateCellNodes{D}) where D = D
Base.size(a::UnivariateCellNodes{D}) where D = (D,)
Base.IndexStyle(::Type{<:UnivariateCellNodes}) = IndexLinear()

function Base.getindex(a::UnivariateCellNodes,d::Integer)
  partition = Tuple(a.partition[d])
  Geometry.CartesianCellNodes(partition)
end

struct UnivariateGrid{D,T,F} <: AbstractVector{Grid{1,1}}
  node_coords::UnivariateCoordinates{D,T,F}
  cell_node_ids::UnivariateCellNodes{D}
  cell_type::Vector{Fill{Int8,1,Tuple{Base.OneTo{Int}}}}

  function UnivariateGrid(desc::UnivariateDescriptor{D,T,F}) where {D,T,F}
    node_coords = UnivariateCoordinates(desc)
    cell_node_ids = UnivariateCellNodes(desc.partition)
    cell_type = map(x->Fill(Int8(1),length(x)),cell_node_ids)
    new{D,T,F}(node_coords,cell_node_ids,cell_type)
  end
end

Base.length(g::UnivariateGrid{D}) where D = D
Base.size(g::UnivariateGrid{D}) where D = (D,)
Base.IndexStyle(::Type{<:UnivariateGrid}) = IndexLinear()

function Base.getindex(g::UnivariateGrid{D},d::Integer) where D
  desc = get_cartesian_descriptor(g)
  CartesianGrid(desc[d])
end

Geometry.OrientationStyle(a::UnivariateGrid) = Oriented()

Geometry.is_regular(a::UnivariateGrid) = is_regular(a[1])

function Geometry.get_cartesian_descriptor(a::UnivariateGrid)
  a.node_coords.data
end

Geometry.get_node_coordinates(g::UnivariateGrid) = g.node_coords

Geometry.get_cell_type(g::UnivariateGrid) = g.cell_type

Geometry.get_cell_node_ids(g::UnivariateGrid) = g.cell_node_ids

function Geometry.get_cell_map(g::UnivariateGrid{D,T,typeof(identity)} where {D,T})
  UnivariateMap(get_cartesian_descriptor(g))
end

function Geometry.get_reffes(g::UnivariateGrid{D}) where D
  ntuple(i->[SEG2,],D)
end

struct UnivariateMap{D,T,L} <: AbstractVector{Geometry.CartesianMap{D,T,L}}
  data::UnivariateDescriptor{D,T,typeof(identity)}
  function UnivariateMap(des::UnivariateDescriptor{D,T}) where {D,T}
    L = D*D
    new{D,T,L}(des)
  end
end

Base.length(a::UnivariateMap{D}) where D = D
Base.size(a::UnivariateMap{D}) where D = (D,)
Base.IndexStyle(::Type{<:UnivariateMap}) = IndexLinear()

function Base.getindex(a::UnivariateMap,d::Integer)
  data = a.data[d]
  CartesianMap(data)
end

function Arrays.lazy_map(::typeof(∇),a::UnivariateMap)
  @notimplemented
end

function Arrays.lazy_map(::typeof(∇),a::LazyArray{<:Fill{<:Reindex{<:UnivariateMap}}})
  @notimplemented
end

struct UnivariateDiscreteModel{D,T,F} <: AbstractVector{DiscreteModel{1,1}}
  grid::UnivariateGrid{D,T,F}
  grid_topology::Vector{UnstructuredGridTopology{1,1,T,Oriented}}
  face_labeling::Vector{FaceLabeling}
end

function UnivariateDiscreteModel(desc::UnivariateDescriptor{D,T,F}) where {D,T,F}
  if any(desc.isperiodic)
    @notimplemented "Have not yet implemented periodic boundary conditions"
  end
  grid = UnivariateGrid(desc)
  _grid = UnstructuredGrid(grid)
  topo,labels = map(_grid) do g
    topo = UnstructuredGridTopology(g)
    nfaces = [num_faces(topo,d) for d in 0:num_cell_dims(topo)]
    labels = FaceLabeling(nfaces)
    Geometry._fill_cartesian_face_labeling!(labels,topo)
    topo,labels
  end |> tuple_of_arrays
  return UnivariateDiscreteModel(grid,topo,labels)
end

function UnivariateDiscreteModel(args...; kwargs...)
  desc = UnivariateDescriptor(args...; kwargs...)
  UnivariateDiscreteModel(desc)
end

function Geometry.UnstructuredGrid(grid::UnivariateGrid{D}) where D
  @assert is_regular(grid) "UnstructuredGrid constructor only for regular grids"
  node_coordinates = collect1d.(get_node_coordinates(grid))
  cell_node_ids = Table.(get_cell_node_ids(grid))
  reffes = get_reffes(grid)
  cell_types = collect1d.(get_cell_type(grid))
  orien = fill(OrientationStyle(grid),D)
  map(UnstructuredGrid,node_coordinates,cell_node_ids,reffes,cell_types,orien)
end

Base.length(model::UnivariateDiscreteModel{D}) where D = D
Base.size(model::UnivariateDiscreteModel{D}) where D = (D,)

# function Base.getindex(model::UnivariateDiscreteModel,d::Integer)
#   CartesianDiscreteModel(grid,grid_topology,face_labeling)
# end

# function Geometry.get_cartesian_descriptor(model::CartesianDiscreteModel)
#   get_cartesian_descriptor(model.grid)
# end

# Geometry.get_grid(model::CartesianDiscreteModel) = model.grid

# Geometry.get_grid_topology(model::CartesianDiscreteModel) = model.grid_topology

# Geometry.get_face_labeling(model::CartesianDiscreteModel) = model.face_labeling

# function Geometry.get_face_nodes(model::CartesianDiscreteModel,d::Integer)
#   face_nodes::Table{Int32,Vector{Int32},Vector{Int32}} = Table(compute_face_nodes(model,d))
#   face_nodes
# end

# function Geometry.get_face_type(model::CartesianDiscreteModel,d::Integer)
#   _, face_to_ftype::Vector{Int8} = compute_reffaces(ReferenceFE{d},model)
#   face_to_ftype
# end

# function Geometry.get_reffaces(::Type{ReferenceFE{d}},model::CartesianDiscreteModel) where d
#   reffaces::Vector{LagrangianRefFE{d}},_ = compute_reffaces(ReferenceFE{d},model)
#   reffaces
# end
