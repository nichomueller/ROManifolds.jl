function Arrays.lazy_map(k::Reindex{<:Table},::Type{T},j_to_i::AbstractArray) where T
  i_to_v = k.values
  Table(i_to_v[j_to_i])
end

function Arrays.lazy_map(k::Reindex{<:CompressedArray},::Type{T},j_to_i::AbstractArray) where T
  i_to_v = k.values
  values = i_to_v.values
  ptrs = i_to_v.ptrs[j_to_i]
  CompressedArray(values,ptrs)
end

function Arrays.lazy_map(k::Reindex{<:PTArray},j_to_i::AbstractArray)
  map(value -> lazy_map(Reindex(value),j_to_i),k.values)
end

struct ReducedUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
  node_coordinates::Vector{Point{Dp,Tp}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map
  function ReducedUnstructuredGrid(
    parent::UnstructuredGrid{Dc,Dp,Tp,O,Tn},
    cell_to_parent_cell::AbstractArray) where {Dc,Dp,Tp,O,Tn}

    node_coordinates = get_node_coordinates(parent)
    cell_node_ids = Table(get_cell_node_ids(parent)[cell_to_parent_cell])
    reffes = get_reffes(parent)
    cell_types = get_cell_type(parent)[cell_to_parent_cell]
    orientation_style = parent.orientation_style
    facet_normal = if Tn == Nothing
      nothing
    else
      lazy_map(Reindex(get_facet_normal(parent)),cell_to_parent_cell)
    end
    cell_map = lazy_map(Reindex(get_cell_map(parent)),cell_to_parent_cell)
    new{Dc,Dp,Tp,O,Tn}(node_coordinates,cell_node_ids,reffes,cell_types,
      orientation_style,facet_normal,cell_map)
  end
end

Geometry.get_reffes(g::ReducedUnstructuredGrid) = g.reffes
Geometry.get_cell_type(g::ReducedUnstructuredGrid) = g.cell_types
Geometry.get_node_coordinates(g::ReducedUnstructuredGrid) = g.node_coordinates
Geometry.get_cell_node_ids(g::ReducedUnstructuredGrid) = g.cell_node_ids
Geometry.get_cell_map(g::ReducedUnstructuredGrid) = g.cell_map
function Geometry.get_facet_normal(g::ReducedUnstructuredGrid)
  @assert !isnothing(g.facet_normal) "This Grid does not have information about normals."
  g.facet_normal
end

abstract type ReducedTriangulation{Dt,Dp} <: Triangulation{Dt,Dp} end

function ReducedTriangulation(::Triangulation,::AbstractArray)
  @notimplemented
end

function is_parent(parent::Triangulation,child::Triangulation;kwargs...)
  false
end

function Geometry.get_glue(trian::ReducedTriangulation{Dt},::Val{Dt}) where Dt
  tface_to_mface = trian.tface_to_mface
  tface_to_mface_map = Fill(GenericField(identity),num_cells(trian))
  if isa(tface_to_mface,IdentityVector) && num_faces(trian.model,Dt) == num_cells(trian)
    mface_to_tface = tface_to_mface
  else
    nmfaces = num_faces(trian.model,Dt)
    mface_to_tface = PosNegPartition(tface_to_mface,Int32(nmfaces))
  end
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
end

function CellData.change_domain(
  a::CellField,
  strian::Triangulation,::ReferenceDomain,
  ttrian::ReducedTriangulation,::ReferenceDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if is_parent(strian,ttrian)
    sface_to_field = get_data(a)
    sglue = get_glue(strian,Val(D))
    mface_to_sface = sglue.mface_to_tface
    tface_to_mface = ttrian.tface_to_mface
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
    return similar_cell_field(a,tface_to_field_s,ttrian,ReferenceDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  change_domain_ref_ref(a,ttrian,sglue,tglue)
end

function CellData.change_domain(
  a::CellDof,
  strian::Triangulation,::ReferenceDomain,
  ttrian::ReducedTriangulation,::ReferenceDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if is_parent(strian,ttrian)
    return CellDof(ttrian.tface_to_mface,ttrian,ReferenceDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  change_domain_ref_ref(a,ttrian,sglue,tglue)
end

function CellData.change_domain(
  a::CellField,
  strian::Triangulation,::PhysicalDomain,
  ttrian::ReducedTriangulation,::PhysicalDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if is_parent(strian,ttrian)
    sface_to_field = get_data(a)
    sglue = get_glue(strian,Val(D))
    mface_to_sface = sglue.mface_to_tface
    tface_to_mface = ttrian.tface_to_mface
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
    return similar_cell_field(a,tface_to_field_s,ttrian,PhysicalDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  change_domain_phys_phys(a,ttrian,sglue,tglue)
end

function CellData.change_domain(
  a::CellDof,
  strian::Triangulation,::PhysicalDomain,
  ttrian::ReducedTriangulation,::PhysicalDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if is_parent(strian,ttrian)
    return CellDof(ttrian.tface_to_mface,ttrian,PhysicalDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  change_domain_phys_phys(a,ttrian,sglue,tglue)
end

function ReducedMeasure(meas::Measure,trians::Triangulation...)
  trian = get_triangulation(meas)
  for t in trians
    if is_parent(t,trian;shallow=true)
      new_trian = ReducedTriangulation(trian,t)
      @unpack (cell_quad,cell_point,cell_weight,trian,
        data_domain_style,integration_domain_style) = meas.quad
      new_quad = CellQuadrature(cell_quad,cell_point,cell_weight,new_trian,
        data_domain_style,integration_domain_style)
      new_meas = Measure(new_quad)
      return new_meas
    end
  end
  @unreachable
end

struct ReducedBodyFittedTriangulation{Dt,Dp,A,B,C} <: ReducedTriangulation{Dt,Dp}
  parent_id::UInt
  model::A
  grid::B
  tface_to_mface::C
  function ReducedBodyFittedTriangulation(
    parent_id::UInt,model::DiscreteModel,grid::Grid,tface_to_mface)
    Dp = num_point_dims(model)
    @assert Dp == num_point_dims(grid)
    Dt = num_cell_dims(grid)
    A = typeof(model)
    B = typeof(grid)
    C = typeof(tface_to_mface)
    new{Dt,Dp,A,B,C}(parent_id,model,grid,tface_to_mface)
  end
end

function ReducedTriangulation(trian::BodyFittedTriangulation,ids::AbstractArray)
  parent_id = objectid(parent)
  model = trian.model
  grid = ReducedUnstructuredGrid(trian.grid,ids)
  tface_to_mface = trian.tface_to_mface[ids]
  ReducedBodyFittedTriangulation(parent_id,model,grid,tface_to_mface)
end

function ReducedTriangulation(child::ReducedBodyFittedTriangulation,parent::BodyFittedTriangulation)
  parent_id = objectid(parent)
  ReducedBodyFittedTriangulation(parent_id,child.model,child.grid,child.tface_to_mface)
end

Geometry.get_background_model(trian::ReducedBodyFittedTriangulation) = trian.model
Geometry.get_grid(trian::ReducedBodyFittedTriangulation) = trian.grid

function is_parent(
  parent::BodyFittedTriangulation{Dc,Dp,Tp,O,Tn},
  child::ReducedBodyFittedTriangulation{Dc,Dp,Tp,O,Tn};
  shallow=false) where {Dc,Dp,Tp,O,Tn}

  if shallow
    get_node_coordinates(get_grid(parent)) == get_node_coordinates(get_grid(child))
  else
    objectid(parent) == child.parent_id
  end
end

struct ReducedBoundaryTriangulation{Dc,Dp,A,B} <: ReducedTriangulation{Dc,Dp}
  parent_id::UInt
  trian::A
  glue::B

  function ReducedBoundaryTriangulation(
    parent_id::UInt,
    trian::BodyFittedTriangulation,
    glue::FaceToCellGlue)

    Dc = num_cell_dims(trian)
    Dp = num_point_dims(trian)
    A = typeof(trian)
    B = typeof(glue)
    new{Dc,Dp,A,B}(parent_id,trian,glue)
  end
end

function ReducedTriangulation(trian::BoundaryTriangulation,ids::AbstractArray)
  parent_id = objectid(trian)

  face_to_bgface = trian.glue.face_to_bgface[ids]
  bgface_to_lcell = trian.glue.bgface_to_lcell

  model = get_background_model(trian)
  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  bgface_grid = Grid(ReferenceFE{D-1},model)

  face_grid = ReducedUnstructuredGrid(bgface_grid,face_to_bgface)
  cell_grid = get_grid(model)
  glue = FaceToCellGlue(topo,cell_grid,face_grid,face_to_bgface,bgface_to_lcell)
  trian = BodyFittedTriangulation(model,face_grid,face_to_bgface)

  ReducedBoundaryTriangulation(parent_id,trian,glue)
end

function ReducedTriangulation(child::ReducedBoundaryTriangulation,parent::BoundaryTriangulation)
  parent_id = objectid(parent)
  ReducedBoundaryTriangulation(parent_id,child.trian,child.glue)
end

Geometry.get_background_model(trian::ReducedBoundaryTriangulation) = get_background_model(trian.trian)
Geometry.get_grid(trian::ReducedBoundaryTriangulation) = get_grid(trian.trian)
Geometry.get_glue(trian::ReducedBoundaryTriangulation{D},::Val{D}) where D = get_glue(trian.trian,Val(D))

function Geometry.get_glue(trian::ReducedBoundaryTriangulation,::Val{Dp}) where Dp
  model = get_background_model(trian)
  Dm = num_cell_dims(model)
  get_glue(trian,Val(Dp),Val(Dm))
end

function Geometry.get_glue(trian::ReducedBoundaryTriangulation,::Val{Dp},::Val{Dm}) where {Dp,Dm}
  nothing
end

function Geometry.get_glue(trian::ReducedBoundaryTriangulation,::Val{D},::Val{D}) where D
  tface_to_mface = trian.glue.face_to_cell
  face_to_q_vertex_coords = _compute_face_to_q_vertex_coords(trian)
  f(p) = get_shapefuns(LagrangianRefFE(Float64,get_polytope(p),1))
  ftype_to_shapefuns = map( f, get_reffes(trian) )
  face_to_shapefuns = expand_cell_data(ftype_to_shapefuns,trian.glue.face_to_ftype)
  face_s_q = lazy_map(linear_combination,face_to_q_vertex_coords,face_to_shapefuns)
  tface_to_mface_map = face_s_q
  mface_to_tface = nothing
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
end

function Geometry.get_facet_normal(trian::ReducedBoundaryTriangulation)
  glue = trian.glue
  cell_grid = get_grid(get_background_model(trian.trian))

  ## Reference normal
  function f(r)
    p = get_polytope(r)
    lface_to_n = get_facet_normal(p)
    lface_to_pindex_to_perm = get_face_vertex_permutations(p,num_cell_dims(p)-1)
    nlfaces = length(lface_to_n)
    lface_pindex_to_n = [fill(lface_to_n[lface],
      length(lface_to_pindex_to_perm[lface])) for lface in 1:nlfaces]
    lface_pindex_to_n
  end
  ctype_lface_pindex_to_nref = map(f, get_reffes(cell_grid))
  face_to_nref = Geometry.FaceCompressedVector(ctype_lface_pindex_to_nref,glue)
  face_s_nref = lazy_map(constant_field,face_to_nref)

  # Inverse of the Jacobian transpose
  cell_q_x = get_cell_map(cell_grid)
  cell_q_Jt = lazy_map(∇,cell_q_x)
  cell_q_invJt = lazy_map(Operation(pinvJt),cell_q_Jt)
  face_q_invJt = lazy_map(Reindex(cell_q_invJt),glue.face_to_cell)

  # Change of domain
  D = num_cell_dims(cell_grid)
  glue = get_glue(trian,Val(D))
  face_s_q = glue.tface_to_mface_map
  face_s_invJt = lazy_map(∘,face_q_invJt,face_s_q)
  face_s_n = lazy_map(Broadcasting(Operation(push_normal)),face_s_invJt,face_s_nref)
  Fields.MemoArray(face_s_n)
end

function Geometry._compute_face_to_q_vertex_coords(trian::ReducedBoundaryTriangulation)
  d = num_cell_dims(trian)
  cell_grid = get_grid(get_background_model(trian.trian))
  polytopes = map(get_polytope, get_reffes(cell_grid))
  ctype_to_lvertex_to_qcoords = map(get_vertex_coordinates, polytopes)
  ctype_to_lface_to_lvertices = map((p)->get_faces(p,d,0), polytopes)
  ctype_to_lface_to_pindex_to_perm = map( (p)->get_face_vertex_permutations(p,d), polytopes)

  P = eltype(eltype(ctype_to_lvertex_to_qcoords))
  D = num_components(P)
  T = eltype(P)
  ctype_to_lface_to_pindex_to_qcoords = Vector{Vector{Vector{Point{D,T}}}}[]

  for (ctype, lface_to_pindex_to_perm) in enumerate(ctype_to_lface_to_pindex_to_perm)
    lvertex_to_qcoods = ctype_to_lvertex_to_qcoords[ctype]
    lface_to_pindex_to_qcoords = Vector{Vector{Point{D,T}}}[]
    for (lface, pindex_to_perm) in enumerate(lface_to_pindex_to_perm)
      cfvertex_to_lvertex = ctype_to_lface_to_lvertices[ctype][lface]
      nfvertices = length(cfvertex_to_lvertex)
      pindex_to_qcoords = Vector{Vector{Point{D,T}}}(undef,length(pindex_to_perm))
      for (pindex, cfvertex_to_ffvertex) in enumerate(pindex_to_perm)
        ffvertex_to_qcoords = zeros(Point{D,T},nfvertices)
        for (cfvertex, ffvertex) in enumerate(cfvertex_to_ffvertex)
          lvertex = cfvertex_to_lvertex[cfvertex]
          qcoords = lvertex_to_qcoods[lvertex]
          ffvertex_to_qcoords[ffvertex] = qcoords
        end
        pindex_to_qcoords[pindex] = ffvertex_to_qcoords
      end
      push!(lface_to_pindex_to_qcoords,pindex_to_qcoords)
    end
    push!(ctype_to_lface_to_pindex_to_qcoords,lface_to_pindex_to_qcoords)
  end

  Geometry.FaceCompressedVector(ctype_to_lface_to_pindex_to_qcoords,trian.glue)
end

function is_parent(
  parent::BoundaryTriangulation{Dc,Dp,A,B},
  child::ReducedBoundaryTriangulation{Dc,Dp,A,B};
  shallow=false) where {Dc,Dp,A,B}

  if shallow
    get_node_coordinates(get_grid(parent)) == get_node_coordinates(get_grid(child))
  else
    objectid(parent) == child.parent_id
  end
end
