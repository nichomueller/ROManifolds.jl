"""
    TProductModel{D,A,B} <: DiscreteModel{D,D} end

Tensor product discrete model, storing a vector of 1-D models `models_1d` of length D,
and the D-dimensional model `model` defined as their tensor product.

"""

struct TProductModel{D,A<:CartesianDiscreteModel{D},B<:AbstractVector{<:CartesianDiscreteModel}} <: DiscreteModel{D,D}
  model::A
  models_1d::B
end

Geometry.get_grid(model::TProductModel) = get_grid(model.model)
Geometry.get_grid_topology(model::TProductModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::TProductModel) = get_face_labeling(model.model)

get_model(model::TProductModel) = model.model
get_1d_models(model::TProductModel) = model.models_1d

function _split_cartesian_descriptor(desc::CartesianDescriptor{D}) where D
  origin,sizes,partition,cmap,isperiodic = desc.origin,desc.sizes,desc.partition,desc.map,desc.isperiodic
  function _compute_1d_desc(
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic))
    CartesianDescriptor(Point(o),(s,),(p,);map=m,isperiodic=(i,))
  end
  descs = map(_compute_1d_desc,origin.data,sizes,partition,Fill(cmap,D),isperiodic)
  return descs
end

function TProductModel(args...;kwargs...)
  desc = CartesianDescriptor(args...;kwargs...)
  descs_1d = _split_cartesian_descriptor(desc)
  model = CartesianDiscreteModel(desc)
  models_1d = CartesianDiscreteModel.(descs_1d)
  TProductModel(model,models_1d)
end

function _axes_to_lower_dim_entities(coords::AbstractArray{VectorValue{D,T},D}) where {D,T}
  function _lower_dim_entities_at_axis!(entities,coords::AbstractArray,ax::Integer)
    range = axes(coords,ax)
    bottom = selectdim(coords,ax,first(range))
    top = selectdim(coords,ax,last(range))
    push!(entities,[bottom,top])
    return
  end
  entities = Vector{Array{VectorValue{D,T},D-1}}[]
  for ax = 1:D
    _lower_dim_entities_at_axis!(entities,coords,ax)
  end
  return entities
end

"""
    entities_1d_in_tag(coords::AbstractArray{VectorValue{D,T},D}, nodes_in_tag
      ) where {D,T} -> (Vector{Int}, Vector{Int})

Given the node coordinates of a D-dimensional tensor product discrete model `coords`
and the subset of nodes in a given tag `nodes_in_tag`, returns the vector of
corresponding 1-D tags, and a vector of axes whose entries are ∈ {1, ..., D}
specify the direction (i.e. dimension) of the tag

"""
function entities_1d_in_tag(coords::AbstractArray{VectorValue{D,T},D},nodes_in_tag) where {D,T}
  ax_to_entities = _axes_to_lower_dim_entities(coords)
  vec_of_tags = Int[]
  vec_of_axes = Int[]
  for (axis,entities) in enumerate(ax_to_entities)
    for (loc,entity) in enumerate(entities)
      Iset = intersect(nodes_in_tag,entity)
      #TODO check_tp_condition(Iset) here i should check that the tags are set correctly
      Iset != entity && continue
      push!(vec_of_tags,loc)
      push!(vec_of_axes,axis)
    end
  end
  return vec_of_tags,vec_of_axes
end

"""
    add_1d_tags!(model::TProductModel,name) -> Nothing

Adds the tags corresponding to `name` (usually a String or Vector{String}), which
encodes a set of tags on a D-dimensional TProductModel, to the vector of 1-D
models

"""
function add_1d_tags!(model::TProductModel,name)
  isempty(name) && return
  nodes = get_node_coordinates(model)
  labeling = get_face_labeling(model)
  face_to_tag = get_face_tag_index(labeling,name,0)

  nodes_in_tag = nodes[findall(face_to_tag.==one(Int8))]
  tags_1d,axs = entities_1d_in_tag(nodes,nodes_in_tag)
  for ax in unique(axs)
    label1d = get_face_labeling(model.models_1d[ax])
    name in label1d.tag_to_name && continue
    tags_at_ax = tags_1d[findall(axs.==ax)]
    add_tag_from_tags!(label1d,name,tags_at_ax)
  end
end

function add_1d_tags!(model::TProductModel,names::AbstractVector)
  map(name->add_1d_tags!(model,name),names)
end

"""
    TProductTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}

Tensor product triangulation, storing a tensor product model, a vector of 1-D
triangulations `trians_1d` of length D, and the D-dimensional triangulation `trian`
defined as their tensor product.

"""
struct TProductTriangulation{Dt,Dp,A<:TProductModel,B<:BodyFittedTriangulation{Dt,Dp},C<:AbstractVector{<:Triangulation}} <: Triangulation{Dt,Dp}
  model::A
  trian::B
  trians_1d::C
end

function TProductTriangulation(trian::Triangulation,trians_1d::AbstractVector{<:Triangulation})
  model = get_background_model(trian)
  models_1d = map(get_background_model,trians_1d)
  tpmodel = TProductModel(model,models_1d)
  TProductTriangulation(tpmodel,trian,trians_1d)
end

Geometry.get_background_model(trian::TProductTriangulation) = trian.model
Geometry.get_grid(trian::TProductTriangulation) = get_grid(trian.trian)
Geometry.get_glue(trian::TProductTriangulation{Dt},::Val{Dt}) where Dt = get_glue(trian.trian,Dt)

function Geometry.Triangulation(model::TProductModel;kwargs...)
  trian = Triangulation(model.model;kwargs...)
  trians_1d = map(Triangulation,model.models_1d)
  TProductTriangulation(model,trian,trians_1d)
end

function Geometry.BoundaryTriangulation(model::TProductModel,args...;kwargs...)
  BoundaryTriangulation(model.model,args...;kwargs...)
end

function CellData.get_cell_points(trian::TProductTriangulation)
  point = get_cell_points(trian.trian)
  single_points = map(get_cell_points,trian.trians_1d)
  TProductCellPoint(point,single_points)
end

struct TProductMeasure{A,B} <: Measure
  measure::A
  measures_1d::B
end

function CellData.Measure(a::TProductTriangulation,args...;kwargs...)
  measure = Measure(a.trian,args...;kwargs...)
  measures_1d = map(Ω -> Measure(Ω,args...;kwargs...),a.trians_1d)
  TProductMeasure(measure,measures_1d)
end

function CellData.get_cell_points(a::TProductMeasure)
  point = get_cell_points(a.measure)
  single_points = map(get_cell_points,a.measures_1d)
  TProductCellPoint(point,single_points)
end

# default behavior

function CellData.integrate(f::CellField,b::TProductMeasure)
  integrate(f,b.measure)
end
