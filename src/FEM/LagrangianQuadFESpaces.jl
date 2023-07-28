abstract type LagrangianQuadRefFE{D} <: Gridap.ReferenceFEs.LagrangianRefFE{D} end
struct LagrangianQuad <: Gridap.ReferenceFEs.ReferenceFEName end
const lagrangian_quad = LagrangianQuad()

function Gridap.ReferenceFEs.ReferenceFE(
  polytope::Polytope,
  ::LagrangianQuad,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}}) where T
  LagrangianQuadRefFE(T,polytope,orders)
end

function LagrangianQuadRefFE(
  ::Type{T},
  p::Polytope{D},
  order::Int) where {T,D}
  orders = Gridap.FESpaces.tfill(order,Val{D}())
  LagrangianQuadRefFE(T,p,orders)
end

function LagrangianQuadRefFE(
  ::Type{T},
  p::Polytope{D},
  orders) where {T,D}
  _lagrangian_quad_ref_fe(T,p,orders)
end

function _lagrangian_quad_ref_fe(
  ::Type{T},
  p::Polytope{D},
  orders) where {T,D}

  @assert all(orders .== first(orders))

  q = Quadrature(p,2*last(orders))
  nodes = get_coordinates(q)

  prebasis = compute_quad_monomial_basis(T,p,orders)

  face_nodes = [Int[] for _ in 1:num_faces(p)]
  push!(last(face_nodes),collect(1:length(nodes))...)

  face_dofs = [Int[] for _ in 1:num_faces(p)]
  push!(last(face_dofs),collect(1:length(nodes)*num_components(T))...)

  dofs = LagrangianDofBasis(T,nodes)

  ndofs = length(dofs.dof_to_node)
  metadata = nothing

  conf = L2Conformity()
  reffe = GenericRefFE{typeof(conf)}(
    ndofs,
    p,
    prebasis,
    dofs,
    conf,
    metadata,
    face_dofs)
  GenericLagrangianRefFE(reffe,face_nodes)
end

function compute_quad_monomial_basis(
  ::Type{T},
  p::ExtrusionPolytope{D},
  orders) where {D,T}

  if D == 0
    MonomialBasis{D}(T,orders,MonomialBasis{D}(T,orders,[CartesianIndex(())]))
  else
    if p in (TRI,QUAD,HEX)
      MonomialBasis{D}(T,orders)
    else p == TET
      order = first(orders)
      if order > 2
        MonomialBasis{D}(T,orders)
      else
        terms = _quad_monomial_terms(order)
        MonomialBasis{D}(T,orders,terms)
      end
    end
  end
end

function _quad_monomial_terms(order)
  if order == 1
    [CartesianIndex((1,1,1)),CartesianIndex((2,1,1)),CartesianIndex((1,2,1)),
    CartesianIndex((1,1,2))]
  else order == 2
    [CartesianIndex((1,1,1)),CartesianIndex((2,1,1)),CartesianIndex((3,1,1)),
    CartesianIndex((1,2,1)),CartesianIndex((2,2,1)),CartesianIndex((1,3,1)),
    CartesianIndex((1,1,2)),CartesianIndex((2,1,2)),CartesianIndex((1,2,2)),
    CartesianIndex((1,1,3)),CartesianIndex((2,2,2))]
  end
end

function _add_quad_terms!(terms,term,extrusion,orders,dim)
  _term = copy(term)
  _orders = copy(orders)
  indexbase = 1
  for i in 0:_orders[dim]
    _term[dim] = i + indexbase
    if dim > 1
      if (extrusion[dim] == TET_AXIS) && i != 0
        _orders .-= 1
      end
      _add_quad_terms!(terms,_term,extrusion,_orders,dim-1)
    else
      push!(terms,CartesianIndex(Tuple(_term)))
    end
  end
end

function Gridap.get_background_model(test::SingleFieldFESpace)
  get_background_model(get_triangulation(test))
end

function get_dimension(::UnstructuredDiscreteModel{Dc,Dp,Tp,B}) where {Dc,Dp,Tp,B}
  Dp
end

function get_dimension(test::SingleFieldFESpace)
  model = get_background_model(test)
  get_dimension(model)
end

function get_cell_points(test::SingleFieldFESpace)
  cell_quad = CellQuadrature(get_triangulation(test),get_degree(test))
  get_data(get_cell_points(cell_quad))
end

struct LagrangianQuadFESpace
  test::SingleFieldFESpace

  function LagrangianQuadFESpace(model::DiscreteModel,order::Int)
    reffe_quad = Gridap.ReferenceFE(lagrangian_quad,Float,order)
    test = TestFESpace(model,reffe_quad,conformity=:L2)
    new(test)
  end
end

function LagrangianQuadFESpace(test::SingleFieldFESpace)
  model = get_background_model(test)
  order = Gridap.FESpaces.get_order(test)
  LagrangianQuadFESpace(model,order)
end

function get_phys_quad_points(
  test::SingleFieldFESpace;
  cells=eachindex(test.cell_dofs_ids))

  trian = get_triangulation(test)
  phys_map = get_cell_map(view(trian,cells))
  cell_points = get_cell_points(test)

  lazy_quadp = map(evaluate,phys_map,cell_points)
  ncells = length(lazy_quadp)
  nquad_cell = length(first(lazy_quadp))

  dim = get_dimension(test)
  quadp = zeros(VectorValue{dim,Float},ncells*nquad_cell)
  @inbounds for (i,quadpi) = enumerate(lazy_quadp)
    quadp[(i-1)*nquad_cell+1:i*nquad_cell] = quadpi
  end

  quadp
end
