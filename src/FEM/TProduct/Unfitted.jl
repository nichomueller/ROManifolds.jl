struct TProductEmbeddedDiscretization{Dp,T} <: GridapEmbedded.Interfaces.AbstractEmbeddedDiscretization
  cutgeo::EmbeddedDiscretization{Dp,T}
end

function GridapEmbedded.cut(cutter::LevelSetCutter,background::TProductModel,geom)
  data = GridapEmbedded.LevelSetCutters._cut_ls(background,geom)
  cutgeo = EmbeddedDiscretization(background,data...,geom)
  TProductEmbeddedDiscretization(cutgeo)
end

function GridapEmbedded.AgFEM.init_bboxes(cell_to_coords,cut::TProductEmbeddedDiscretization;kwargs...)
  GridapEmbedded.AgFEM.init_bboxes(cell_to_coords,cut.cutgeo;kwargs...)
end

function GridapEmbedded.aggregate(strategy,cut::TProductEmbeddedDiscretization,args...)
  aggregate(strategy,cut.cutgeo,args...)
end

function Geometry.get_background_model(cut::TProductEmbeddedDiscretization)
  get_background_model(cut.cutgeo)
end

function GridapEmbedded.get_geometry(cut::TProductEmbeddedDiscretization)
  get_geometry(cut.cutgeo)
end

function GridapEmbedded.compute_bgcell_to_inoutcut(cut::TProductEmbeddedDiscretization,args...)
  compute_bgcell_to_inoutcut(cut.cutgeo,args...)
end

function Geometry.Triangulation(cut::TProductEmbeddedDiscretization)
  Triangulation(cut,PHYSICAL_IN,cut.cutgeo)
end

function Geometry.Triangulation(cut::TProductEmbeddedDiscretization,in_or_out)
  Triangulation(cut,in_or_out,cut.cutgeo.geo)
end

function Geometry.Triangulation(cut::TProductEmbeddedDiscretization,name::String)
  geo = get_geometry(cut.cutgeo.geo,name)
  Triangulation(cut,PHYSICAL_IN,geo)
end

function Geometry.Triangulation(cut::TProductEmbeddedDiscretization,geo::GridapEmbedded.CSG.Geometry)
  Triangulation(cut,PHYSICAL_IN,geo)
end

function Geometry.Triangulation(cut::TProductEmbeddedDiscretization,in_or_out,name::String)
  geo = get_geometry(cut.cutgeo.geo,name)
  Triangulation(cut,in_or_out,geo)
end

function Geometry.Triangulation(
  cut::TProductEmbeddedDiscretization,
  in_or_out::Tuple,
  geo::GridapEmbedded.CSG.Geometry)

  trian1 = Triangulation(cut,in_or_out[1],geo)
  trian2 = Triangulation(cut,in_or_out[2],geo)
  num_cells(trian1) == 0 ? trian2 : lazy_append(trian1,trian2)
end

function Geometry.Triangulation(
  cut::TProductEmbeddedDiscretization,
  in_or_out::GridapEmbedded.Interfaces.CutInOrOut,
  geo::GridapEmbedded.CSG.Geometry)

  Triangulation(cut.cutgeo,in_or_out,geo)
end

function Geometry.Triangulation(
  cut::TProductEmbeddedDiscretization,
  in_or_out::Union{Integer,GridapEmbedded.Interfaces.ActiveInOrOut},
  geo::GridapEmbedded.CSG.Geometry)

  bgmodel = get_background_model(cut)
  cutgeo = GridapEmbedded.Distributed.change_bgmodel(cut.cutgeo,bgmodel.model)
  trian = Triangulation(cutgeo,in_or_out,geo)
  trians_1d = map(Triangulation,bgmodel.models_1d)
  TProductTriangulation(bgmodel,trian,trians_1d)
end

function GridapEmbedded.Interfaces.compute_subcell_to_inout(cut::TProductEmbeddedDiscretization,args...)
  GridapEmbedded.Interfaces.compute_subcell_to_inout(cut.cutgeo,args...)
end

function GridapEmbedded.EmbeddedBoundary(cut::TProductEmbeddedDiscretization,args...)
  bgmodel = get_background_model(cut)
  cutgeo = GridapEmbedded.Distributed.change_bgmodel(cut.cutgeo,bgmodel.model)
  EmbeddedBoundary(cutgeo,args...)
end

function GridapEmbedded.GhostSkeleton(cut::TProductEmbeddedDiscretization,in_or_out,name::String)
  GhostSkeleton(cut.cutgeo,args...)
end

function GridapEmbedded.AgFEMSpace(
  trian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any},
  bgcell_to_bgcellin::AbstractVector,
  args...;
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args

  model = get_background_model(trian)
  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)

  space = FESpace(trian.trian,cell_reffe;kwargs...)
  agspace = AgFEMSpace(space,bgcell_to_bgcellin,args...)
  spaces_1d = univariate_spaces(model,trian,cell_reffes_1d;kwargs...)
  TProductFESpace(agspace,spaces_1d,trian)
end
