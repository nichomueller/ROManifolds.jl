const DistributedContribution{V} = GenericContribution{DistributedTriangulation,V}

const DistributedArrayContribution = DistributedContribution{T} where {T<:AbstractArray}

distributed_array_contribution() = GenericContribution(IdDict{DistributedTriangulation,AbstractArray}())

struct DistributedContributionBroadcast{D}
  contrib::D
end

function Base.broadcasted(f,a::DistributedArrayContribution,b::Number)
  BT = PartitionedArrays.PBroadcasted{<:AbstractArray{<:FEM.ParamBroadcast}}
  c = GenericContribution(IdDict{DistributedTriangulation,BT}())
  for (trian,values) in a.dict
    c[trian] = Base.broadcasted(f,values,b)
  end
  DistributedContributionBroadcast(c)
end

function Base.materialize(c::DistributedContributionBroadcast)
  a = distributed_array_contribution()
  for (trian,b) in c.contrib.dict
    a[trian] = Base.materialize(b)
  end
  a
end

function Base.materialize!(a::DistributedArrayContribution,c::DistributedContributionBroadcast)
  for (trian,b) in c.contrib.dict
    val = a[trian]
    Base.materialize!(val,b)
  end
  a
end

Base.eltype(a::DistributedArrayContribution) = eltype(first(values(a.dict)))
Base.eltype(a::Tuple{Vararg{DistributedArrayContribution}}) = eltype(first(a))

function LinearAlgebra.fillstored!(a::DistributedArrayContribution,v)
  for c in values(a.dict)
    LinearAlgebra.fillstored!(c,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::Tuple{Vararg{DistributedArrayContribution}},v)
  map(a) do a
    LinearAlgebra.fillstored!(a,v)
  end
end

const DistributedAffineContribution = DistributedContribution{T} where {T<:AbstractVector{<:AffineContribution}}

distributed_affine_contribution() = GenericContribution(IdDict{DistributedTriangulation,AffineDecomposition}())
