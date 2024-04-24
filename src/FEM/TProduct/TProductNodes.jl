abstract type IsotropyStyle end
struct Isotropic <: IsotropyStyle end
struct Anisotropic <: IsotropyStyle end

Isotropy(b::Isotropic...) = Isotropic()
Isotropy(b::IsotropyStyle...) = Anisotropic()

Isotropy(b::Bool...) = all(b) ? Isotropic() : Anisotropic()
Isotropy(a::NTuple{D}) where D = Isotropy(all([a[d] == a[1] for d = 1:D]))
Isotropy(a::AbstractVector) = Isotropy(all([a[d] == a[1] for d = eachindex(a)]))
Isotropy(::typeof(identity)) = Isotropy(true)
Isotropy(::Function) = Isotropy(false)

abstract type AbstractTensorProductPoints{D,I,T,N} <: AbstractArray{Point{D,T},N} end

get_factors(::AbstractTensorProductPoints) = @abstractmethod
get_indices_map(::AbstractTensorProductPoints) = @abstractmethod
ReferenceFEs.num_dims(::AbstractTensorProductPoints{D}) where D = D
get_isotropy(::AbstractTensorProductPoints{D,I} where D) where I = I
tensor_product_points(::Type{<:AbstractTensorProductPoints},nodes,::NodesMap) = @abstractmethod

struct TensorProductNodes{D,I,T,A,B} <: AbstractTensorProductPoints{D,I,T,1}
  nodes::A
  indices_map::B
  isotropy::I
  function TensorProductNodes(
    ::Type{T},
    nodes::A,
    indices_map::B,
    isotropy::I=Isotropy(nodes)
    ) where {D,I,T,A,B<:NodesMap{D}}
    new{D,I,T,A,B}(nodes,indices_map,isotropy)
  end
end

function TensorProductNodes(nodes::NTuple{D,AbstractVector{Point{1,T}}},args...) where {D,T}
  TensorProductNodes(T,nodes,args...)
end

function TensorProductNodes(nodes::AbstractVector{<:AbstractVector{Point{1,T}}},args...) where T
  TensorProductNodes(T,nodes,args...)
end

function TensorProductNodes(polytope::Polytope{D},orders) where D
  function _compute_1d_nodes(order=first(orders))
    nodes,_ = compute_nodes(SEGMENT,(order,))
    return nodes
  end
  isotropy = Isotropy(orders)
  nodes = isotropy==Isotropic() ? Fill(_compute_1d_nodes(),D) : map(_compute_1d_nodes,orders)
  indices_map = compute_nodes_map(;polytope,orders)
  TensorProductNodes(nodes,indices_map,isotropy)
end

function TensorProductNodes(polytope::Polytope{D},order::Integer=1) where D
  TensorProductNodes(polytope,tfill(order,Val(D)))
end

get_factors(a::TensorProductNodes) = a.nodes
get_indices_map(a::TensorProductNodes) = a.indices_map

function tensor_product_points(::Type{<:TensorProductNodes},nodes,indices_map::NodesMap)
  TensorProductNodes(nodes,indices_map)
end

ReferenceFEs.num_nodes(a::TensorProductNodes) = num_nodes(get_indices_map(a))

Base.length(a::TensorProductNodes) = num_nodes(a)
Base.size(a::TensorProductNodes) = (num_nodes(a),)
Base.axes(a::TensorProductNodes) = (Base.OneTo(num_nodes(a)),)
Base.IndexStyle(::TensorProductNodes) = IndexLinear()

function Base.getindex(a::TensorProductNodes{D,I,T},i::Integer) where {D,I,T}
  indices_map = get_indices_map(a)
  entry = indices_map[i]
  p = zero(Mutable(Point{D,T}))
  @inbounds for d in 1:D
    p[d] = a.nodes[d][entry[d]][1]
  end
  Point(p)
end

function Base.copy(a::TensorProductNodes{D,I,T}) where {D,I,T}
  TensorProductNodes(T,copy.(a.nodes),a.indices_map,a.isotropy)
end
