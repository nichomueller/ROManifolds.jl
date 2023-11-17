abstract type TriangulationWithTags{Dc,Dp} <: Triangulation{Dc,Dp} end

get_tags(trian::TriangulationWithTags) = trian.tags
get_id(trian::TriangulationWithTags) = trian.id

function Base.:(==)(a::TriangulationWithTags,b::TriangulationWithTags)
  get_id(a) == get_id(b)
end

function TriangulationWithTags(model::DiscreteModel,filter::AbstractArray)
  d = num_cell_dims(model)
  TriangulationWithTagsWithTags(ReferenceFE{d},model,filter)
end

function TriangulationWithTags(
  ::Type{ReferenceFE{d}},
  model::DiscreteModel,
  labels::FaceLabeling;
  tags=nothing,
  id=nothing) where d

  id = isnothing(id) ? rand(UInt) : id
  if isnothing(tags)
    grid = Grid(ReferenceFE{d},model)
    tface_to_mface = IdentityVector(num_cells(grid))
    BodyFittedTriangulationWithTags(model,grid,tface_to_mface,tags,id)
  else
    mface_to_mask = get_face_mask(labels,tags,d)
    mgrid = Grid(ReferenceFE{d},model)
    tgrid = GridPortion(mgrid,mface_to_mask)
    tface_to_mface = tgrid.cell_to_parent_cell
    BodyFittedTriangulationWithTags(model,tgrid,tface_to_mface,tags,id)
  end
end

function TriangulationWithTags(
  ::Type{ReferenceFE{d}},model::DiscreteModel;kwargs...) where d
  labels = get_face_labeling(model)
  TriangulationWithTags(ReferenceFE{d},model,labels;kwargs...)
end

function TriangulationWithTags(model::DiscreteModel;kwargs...)
  d = num_cell_dims(model)
  labels = get_face_labeling(model)
  TriangulationWithTags(ReferenceFE{d},model,labels;kwargs...)
end

function TriangulationWithTags(trian::TriangulationWithTags,args...;kwargs...)
  @notimplemented
end

function InteriorWithTags(args...;kwargs...)
  TriangulationWithTags(args...;kwargs...)
end

struct BodyFittedTriangulationWithTags{Dt,Dp,A,B,C,T} <: TriangulationWithTags{Dt,Dp}
  model::A
  grid::B
  tface_to_mface::C
  tags::T
  id::UInt

  function BodyFittedTriangulationWithTags(
    model::DiscreteModel,
    grid::Grid,tface_to_mface,
    tags::T,
    id::UInt) where T

    Dp = num_point_dims(model)
    @assert Dp == num_point_dims(grid)
    Dt = num_cell_dims(grid)
    A = typeof(model)
    B = typeof(grid)
    C = typeof(tface_to_mface)
    new{Dt,Dp,A,B,C,T}(model,grid,tface_to_mface,tags,id)
  end
end

Geometry.get_background_model(trian::BodyFittedTriangulationWithTags) = trian.model
Geometry.get_grid(trian::BodyFittedTriangulationWithTags) = trian.grid

function FESpaces.get_triangulation(trian::BodyFittedTriangulationWithTags)
  model = get_background_model(trian)
  grid = get_grid(trian)
  BodyFittedTriangulation(model,grid,trian.tface_to_mface)
end

function Geometry.get_glue(trian::BodyFittedTriangulationWithTags{Dt},::Val{Dt}) where Dt
  get_glue(get_triangulation(trian),Val(Dt))
end

struct BoundaryTriangulationWithTags{Dc,Dp,A,B,T} <: TriangulationWithTags{Dc,Dp}
  trian::A
  glue::B
  tags::T
  id::UInt

  function BoundaryTriangulationWithTags(
    trian::BodyFittedTriangulation,
    glue::FaceToCellGlue,
    tags::T,
    id::UInt) where T

    Dc = num_cell_dims(trian)
    Dp = num_point_dims(trian)
    A = typeof(trian)
    B = typeof(glue)
    new{Dc,Dp,A,B,T}(trian,glue,tags,id)
  end
end

function BoundaryWithTags(args...;kwargs...)
  BoundaryTriangulationWithTags(args...;kwargs...)
end

function BoundaryTriangulationWithTags(
  model::DiscreteModel,
  face_to_bgface::AbstractVector{<:Integer},
  bgface_to_lcell::AbstractVector{<:Integer},
  tags::T,
  id::UInt) where T

  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  bgface_grid = Grid(ReferenceFE{D-1},model)

  face_grid = view(bgface_grid,face_to_bgface)
  cell_grid = get_grid(model)
  glue = FaceToCellGlue(topo,cell_grid,face_grid,face_to_bgface,bgface_to_lcell)
  trian = BodyFittedTriangulation(model,face_grid,face_to_bgface)

  BoundaryTriangulationWithTags(trian,glue,tags,id)
end

function BoundaryTriangulationWithTags(
  model::DiscreteModel,
  bgface_to_mask::AbstractVector{Bool},
  bgface_to_lcell::AbstractVector{<:Integer},
  tags::T,
  id::UInt) where T

  face_to_bgface = findall(bgface_to_mask)
  BoundaryTriangulationWithTags(model,face_to_bgface,bgface_to_lcell,tags,id)
end

function BoundaryTriangulationWithTags(
  model::DiscreteModel,
  bgface_to_mask::AbstractVector{Bool},
  lcell::Integer=1,
  args...)

  BoundaryTriangulationWithTags(model,bgface_to_mask,Fill(lcell,num_facets(model)),args...)
end

function BoundaryTriangulationWithTags(
  model::DiscreteModel,
  face_to_bgface::AbstractVector{<:Integer},
  args...)

  BoundaryTriangulationWithTags(model,face_to_bgface,Fill(1,num_facets(model)),args...)
end

function BoundaryTriangulationWithTags(
  model::DiscreteModel,
  labels::FaceLabeling;
  tags=nothing,
  id=nothing)

  id = isnothing(id) ? rand(UInt) : id
  D = num_cell_dims(model)
  if isnothing(tags)
    topo = get_grid_topology(model)
    face_to_mask = get_isboundary_face(topo,D-1)
  else
    face_to_mask = get_face_mask(labels,tags,D-1)
  end
  BoundaryTriangulationWithTags(model,face_to_mask,tags,id)
end

function BoundaryTriangulationWithTags(model::DiscreteModel;kwargs...)
  labels = get_face_labeling(model)
  BoundaryTriangulationWithTags(model,labels;kwargs...)
end

function BoundaryTriangulationWithTags(rtrian::Triangulation,args...;kwargs...)
  @notimplemented
end

Geometry.get_background_model(t::BoundaryTriangulationWithTags) = get_background_model(t.trian)
Geometry.get_grid(t::BoundaryTriangulationWithTags) = get_grid(t.trian)

function FESpaces.get_triangulation(t::BoundaryTriangulationWithTags)
  BoundaryTriangulation(t.trian,t.glue)
end

function Geometry.get_glue(trian::BoundaryTriangulationWithTags,::Val{Dp}) where Dp
  get_glue(get_triangulation(trian),Val(Dp))
end

function Geometry.get_glue(::BoundaryTriangulationWithTags,::Val{Dp},::Val{Dm}) where {Dp,Dm}
  nothing
end

function Geometry.get_glue(trian::BoundaryTriangulationWithTags,::Val{D},::Val{D}) where D
  get_glue(get_triangulation(trian),Val(D),Val(D))
end

function Geometry.get_facet_normal(t::BoundaryTriangulationWithTags)
  get_facet_normal(get_triangulation(t))
end

function Geometry._compute_face_to_q_vertex_coords(t::BoundaryTriangulationWithTags)
  Geometry._compute_face_to_q_vertex_coords(get_triangulation(t))
end
