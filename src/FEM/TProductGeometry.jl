struct TProductModel{D,A,B} <: DiscreteModel{D,D}
  model::A
  models_1d::B
  function TProductModel(
    model::A,
    models_1d::B
    ) where {D,A<:CartesianDiscreteModel{D},B<:AbstractVector{<:CartesianDiscreteModel}}
    new{D,A,B}(model,models_1d)
  end
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
  isotropy = all([sizes[d] == sizes[1] && partition[d] == partition[1] for d = 1:D])
  factors = isotropy ? Fill(_compute_1d_desc(),D) : map(_compute_1d_desc,origin.data,sizes,partition,Fill(cmap,D),Fill(isperiodic,D))
  return factors
end

function TProductModel(args...;kwargs...)
  desc = CartesianDescriptor(args...;kwargs...)
  descs_1d = _split_cartesian_descriptor(desc)
  model = CartesianDiscreteModel(desc)
  models_1d = CartesianDiscreteModel.(descs_1d)
  TProductModel(model,models_1d)
end

struct TProductTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}
  model::A
  trian::B
  trians_1d::C
  function TProductTriangulation(
    model::A,
    trian::B,
    trians_1d::C
    ) where {Dt,Dp,A<:TProductModel,B<:BodyFittedTriangulation{Dt,Dp},C<:AbstractVector{<:Triangulation}}
    new{Dt,Dp,A,B,C}(model,trian,trians_1d)
  end
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
