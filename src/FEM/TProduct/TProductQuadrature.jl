order_2_degree(o::Int) = 2*o+1
degree_2_order(d::Int) = Int((d-1)/2)

struct TensorProductQuad <: QuadratureName end

const tpquadrature = TensorProductQuad()

function ReferenceFEs.Quadrature(p::Polytope,name::TensorProductQuad,degree::Int;kwargs...)
  degrees = tfill(degree,Val(num_dims(p)))
  Quadrature(p,name,degrees;kwargs...)
end

function ReferenceFEs.Quadrature(p::Polytope,::TensorProductQuad,degrees;T::Type{<:AbstractFloat}=Float64)
  TensorProductQuadrature(p,degrees;T)
end

struct TensorProductQuadrature{D,I,T,A,B,C} <: Quadrature{D,T}
  factors::A
  quad::B
  quad_map::C
  isotropy::I
  function TensorProductQuadrature(
    factors::A,
    quad::B,
    quad_map::C,
    isotropy::I
    ) where {D,I,T,A,B<:Quadrature{D,T},C}

    new{D,I,T,A,B,C}(factors,quad,quad_map,isotropy)
  end
end

function TensorProductQuadrature(polytope::Polytope{D},degrees;T::Type{<:AbstractFloat}=Float64) where D
  function _compute_1d_quad(degree=first(degrees))
    Quadrature(SEGMENT,(degree,);T)
  end
  isotropy = Isotropy(degrees)
  factors = isotropy==Isotropic() ? Fill(_compute_1d_quad(),D) : map(_compute_1d_quad,degrees)
  quad = Quadrature(polytope,tensor_product,degrees;T)
  orders = map(degree_2_order,degrees)
  indices_map = trivial_nodes_map(;polytope,orders)
  TensorProductQuadrature(factors,quad,indices_map,isotropy)
end

get_factors(q::TensorProductQuadrature) = q.factors

# function ReferenceFEs.get_coordinates(q::TensorProductQuadrature)
#   coords = map(get_coordinates,q.factors)
#   TensorProductNodes(coords,q.quad_map)
# end

# function ReferenceFEs.get_weights(q::TensorProductQuadrature)
#   weights = map(get_weights,q.factors)
#   kronecker(weights)
# end

ReferenceFEs.get_coordinates(q::TensorProductQuadrature) = get_coordinates(q.quad)

ReferenceFEs.get_weights(q::TensorProductQuadrature) = get_weights(q.quad)

ReferenceFEs.get_name(q::TensorProductQuadrature) = get_name(q.quad)

ReferenceFEs.num_point_dims(q::TensorProductQuadrature) = num_point_dims(q.quad)

ReferenceFEs.num_dims(q::TensorProductQuadrature) = num_dims(q.quad)

# function CellData.CellQuadrature(
#   trian::TensorProductTriangulation,
#   cell_quad::AbstractVector{<:TensorProductQuadrature},
#   dds::DomainStyle,
#   ids::DomainStyle)

#   D = num_dims(first(cell_quad))
#   factors = map(get_factors,cell_quad)
#   quads = map(get_quad,cell_quad)
#   cfactors = map(CellQuadrature,trian.factors,factors,Fill(dds,D),Fill(ids,D))
#   cquad = CellQuadrature(trian.trian,quads,dds,ids)
#   TensorProductCellQuadrature(cfactors,cquad)
# end
