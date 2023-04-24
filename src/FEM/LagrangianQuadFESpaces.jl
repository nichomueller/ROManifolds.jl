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

  #= @assert isa(p,ExtrusionPolytope)
  @assert is_n_cube(p)
  degrees = broadcast(*,2,orders)
  q = Quadrature(p,Gridap.ReferenceFEs.TensorProduct(),degrees) =#
  @assert isa(p,ExtrusionPolytope)
  @assert is_n_cube(p) || is_simplex(p) "Wrong polytope"
  q = Quadrature(p,2*last(orders))
  nodes = get_coordinates(q)

  prebasis = compute_monomial_basis(T,p,orders)

  # Compute face_own_nodes
  face_nodes = [Int[] for _ in 1:num_faces(p)]
  push!(last(face_nodes),collect(1:length(nodes))...)

  # Compute face_own_dofs
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

get_degree(order::Int,c=2) = c*order

get_degree(test::SingleFieldFESpace,c=2) = get_degree(Gridap.FESpaces.get_order(test),c)

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

function Gridap.FESpaces.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  first(basis.cell_basis.values[1].fields.orders)
end

function get_cell_quadrature(test::SingleFieldFESpace)
  CellQuadrature(get_triangulation(test),get_degree(test))
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
  order = get_dimension(test)-2
  LagrangianQuadFESpace(model,order)
end

function Gridap.FEFunction(
  quad_fespace::LagrangianQuadFESpace,
  vec::AbstractVector)

  FEFunction(quad_fespace.test,vec)
end

function Gridap.FEFunction(
  quad_fespace::LagrangianQuadFESpace,
  mat::AbstractMatrix)

  n -> FEFunction(quad_fespace.test,mat[:,n])
end

function get_phys_quad_points(test::SingleFieldFESpace)
  trian = get_triangulation(test)
  phys_map = get_cell_map(trian)
  cell_quad = get_cell_quadrature(test)
  cell_points = get_data(get_cell_points(cell_quad))
  lazy_quadp = map(evaluate,phys_map,cell_points)

  lazy_quadp = get_lazy_phys_quad_points(op)
  ncells = length(lazy_quadp)
  nquad_cell = length(first(lazy_quadp))
  nquadp = ncells*nquad_cell
  dim = get_dimension(op)
  quadp = zeros(VectorValue{dim,Float},nquadp)
  @inbounds for (i,quadpi) = enumerate(lazy_quadp)
    quadp[(i-1)*nquad_cell+1:i*nquad_cell] = quadpi
  end

  quadp
end
