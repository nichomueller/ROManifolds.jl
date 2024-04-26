function split(desc::CartesianDescriptor{D}) where D
  @unpack origin,sizes,partition,map,isperiodic = desc
  _split_cartesian_descriptor(origin,sizes,partition,map,isperiodic)
end

struct TensorProductDescriptor{I,A,B} <: GridapType
  factors::A
  isotropy::I
end

get_factors(a::TensorProductDescriptor) = a.factors

function TensorProductDescriptor(
  domain,
  partition::NTuple{D};
  map::Function=identity,
  isperiodic::NTuple=tfill(false,Val(D))) where D

  desc = CartesianDescriptor(domain,partition;map,isperiodic)
  factors,isotropy = split(desc)
  TensorProductDescriptor(factors,isotropy)
end

function TensorProductFaceLabels()

end

struct TensorProductDiscreteModel{D,A} <: DiscreteModel{D,D}
  models::A
  face_labeling::TensorProductFaceLabels
  function TensorProductDiscreteModel(models::A,face_labeling::TensorProductFaceLabels
    ) where {A<:AbstractVector{<:CartesianDiscreteModel{1}}}
    D = length(models)
    new{D,A}(models,face_labeling)
  end
end

function TensorProductDiscreteModel(args...;kwargs...)
  desc = TensorProductDescriptor(args...;kwargs...)
  models = map(CartesianDiscreteModel,desc)
  face_labeling = TensorProductFaceLabels(models,args...;kwargs...)
  TensorProductDiscreteModel(models,face_labeling)
end

Geometry.get_grid(a::TensorProductDiscreteModel) = TensorProductGrid(map(get_grid,a.models))
Geometry.get_grid_topology(a::TensorProductDiscreteModel) = TensorProductGridTopology(map(get_grid_topology,a.models))
Geometry.get_face_labeling(a::TensorProductDiscreteModel) = a.face_labeling

function check_tp_boundary(labels,topo,names)
  isempty(names) && return labels
  for name in names
    check_tp_boundary(labels,topo,name)
  end
end

function check_tp_boundary(labels,topo,name::String)
  tag = get_tag_from_name(labels,name)
  entities = check_tp_boundary(labels,topo,tag)
  isempty(entities) && return
  error("With the prescribed boundary conditions, the geometry is not tensor product")
end

function check_tp_boundary(labels,topo,tag::Integer)
  d0,d1 = 0,1
  D = num_cell_dims(topo)
  polytope = first(get_polytopes(topo))
  offsets = get_offsets(polytope)
  vertices = findall(get_face_tag_index(labels,tag,d0))
  vertices_to_entities = labels.d_to_dface_to_entity[d0]
  entities = vertices_to_entities[vertices]
  add_entities = Vector{Int32}[]
  faces = get_faces(grid_topology,d1,d0)
  for face in faces
    inclusions = map(e->e ∈ face,entities)
    if any(inclusions) && !all(inclusions)
      e_not_in_tag = entities[findall(!inclusions)]
      oe = findlast(e_not_in_tag)
    end
  end
  isempty(add_entities) && return

end
