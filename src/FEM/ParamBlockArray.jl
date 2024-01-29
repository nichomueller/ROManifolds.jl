struct ParamBlockArray{T,N,A,L,B} <: AbstractParamContainer{T,N}
  blockarrays::B
  function ParamBlockArray(
    blockarrays::BlockArray{T,N,<:AbstractArray{ParamArray{T,N,A,L}}}
    ) where {T,N,A,L}
    B = typeof(blockarrays)
    new{T,N,A,L,B}(blockarrays)
  end
end

const ParamBlockVector{T,A,L,B} = ParamBlockArray{T,1,A,L,B}
const ParamBlockMatrix{T,A,L,B} = ParamBlockArray{T,2,A,L,B}

@inline function BlockArrays._BlockArray(
  blocks::AbstractArray{<:AbstractParamContainer},
  block_sizes::Vararg{AbstractVector{<:Integer},N}
  ) where N

  blockarrays = BlockArrays._BlockArray(blocks,map(blockedrange,block_sizes))
  ParamBlockArray(blockarrays)
end

function BlockArrays.mortar(blocks::AbstractArray{<:AbstractParamContainer})
  s = BlockArrays.sizes_from_blocks(first.(blocks))
  BlockArrays._BlockArray(blocks,s...)
end

Arrays.get_array(a::ParamBlockArray) = a.blockarrays
Arrays.testitem(a::ParamBlockArray) = first(a)
BlockArrays.blocks(a::ParamBlockArray) = blocks(get_array(a))
BlockArrays.blocklength(a::ParamBlockArray) = blocklength(get_array(a))
BlockArrays.blocksize(a::ParamBlockArray) = blocksize(get_array(a))
Base.length(a::ParamBlockArray) = length(first(blocks(a)))
Base.size(a::ParamBlockArray) = map(length,axes(a))
Base.axes(a::ParamBlockArray) = axes(get_array(a))
Base.eltype(::ParamBlockArray{T}) where T = T
Base.eltype(::Type{<:ParamBlockArray{T}}) where T = T
Base.ndims(::ParamBlockArray{T,N} where T) where N = N
Base.ndims(::Type{<:ParamBlockArray{T,N}} where T) where N = N
Base.eachindex(::ParamBlockArray{T,N,A,L}) where {T,N,A,L} = Base.OneTo(L)
Base.first(a::ParamBlockArray) = getindex(a,1)

function Base.getindex(a::ParamBlockArray,i::Integer)
  blocksi = getindex.(blocks(a),i)
  BlockArrays._BlockArray(blocksi,axes(a))
end

function Base.show(io::IO,::MIME"text/plain",a::ParamBlockArray{T,N,A,L}) where {T,N,A,L}
  s = size(blocks(a))
  _nice_size(s::Tuple{Int}) = "$(s[1]) - "
  _nice_size(s::Tuple{Int,Int}) = "$(s[1]) × $(s[2]) - "
  println(io, _nice_size(s) *"parametric block array of types $(eltype(A)) and length $L, with entries:")
  show(io,a.blockarrays)
end

function Base.copy(a::ParamBlockArray)
  b = map(blocks(a)) do block
    copy(block)
  end
  blockarrays = BlockArrays._BlockArray(b,axes(a))
  ParamBlockArray(blockarrays)
end

function Base.similar(
  a::ParamBlockArray{T},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(first(blocks(a)))) where {T,S}

  b = map(blocks(a)) do block
    similar(block,element_type,dims)
  end
  blockarrays = BlockArrays._BlockArray(b,axes(a))
  ParamBlockArray(blockarrays)
end

function Base.:+(a::T,b::T) where T<:ParamBlockArray
  c = map(blocks(a),blocks(b)) do blocka,blockb
    blocka+blockb
  end
  blockarrays = BlockArrays._BlockArray(c,axes(a))
  ParamBlockArray(blockarrays)
end

function Base.:-(a::T,b::T) where T<:ParamBlockArray
  c = map(blocks(a),blocks(b)) do blocka,blockb
    blocka-blockb
  end
  blockarrays = BlockArrays._BlockArray(c,axes(a))
  ParamBlockArray(blockarrays)
end

function LinearAlgebra.fillstored!(a::ParamBlockMatrix,v)
  map(ai->LinearAlgebra.fillstored!(ai,v),blocks(a))
end

function Base.fill!(a::ParamBlockVector,v)
  map(ai->fill!(ai,v),blocks(a))
end

function LinearAlgebra.mul!(
  c::ParamBlockArray,
  a::ParamBlockArray,
  b::ParamBlockArray,
  α::Number,β::Number)

  @assert length(a) == length(b) == length(c)
  map(blocks(c),blocks(a),blocks(b)) do c,a,b
    mul!(c,a,b,α,β)
  end
  c
end

function LinearAlgebra.ldiv!(a::ParamBlockArray,m::LU,b::ParamBlockArray)
  @assert length(a) == length(b)
  map(blocks(a),blocks(b)) do a,b
    ldiv!(a,m,b)
  end
  a
end

function LinearAlgebra.ldiv!(a::ParamBlockArray,m::AbstractArray,b::ParamBlockArray)
  @assert length(a) == length(m) == length(b)
  map(blocks(a),m,blocks(b)) do a,m,b
    ldiv!(a,m,b)
  end
  a
end

function LinearAlgebra.rmul!(a::ParamBlockArray,b::Number)
  map(a) do blocks(a)
    rmul!(a,b)
  end
end

function LinearAlgebra.lu(a::ParamBlockArray)
  lua = map(a) do blocks(a)
    lu(a)
  end
  ParamContainer(lua)
end

function LinearAlgebra.lu!(a::ParamBlockArray,b::ParamBlockArray)
  @assert length(a) == length(b)
  map(blocks(a),blocks(b)) do a,b
    lu!(a,b)
  end
  a
end

struct ParamBlockBroadcast{D}
  array::D
end

BlockArrays.blocks(a::ParamBlockBroadcast{<:ParamBlockArray}) = a.array

function Base.broadcasted(f,a::Union{ParamBlockArray,ParamBlockBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(blocks,a)...)
  ParamBlockBroadcast(bc)
end

function Base.broadcasted(f,a::Union{ParamBlockArray,ParamBlockBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),blocks(a))
  ParamBlockBroadcast(bc)
end

function Base.broadcasted(f,a::Number,b::Union{ParamBlockArray,ParamBlockBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),blocks(b))
  ParamBlockBroadcast(bc)
end

function Base.broadcasted(f,
  a::Union{ParamBlockArray,ParamBlockBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{ParamBlockArray,ParamBlockBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::ParamBlockBroadcast)
  a = map(Base.materialize,blocks(b))
  mortar(a)
end

function Base.materialize!(a::ParamBlockArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),blocks(a))
  a
end

function Base.materialize!(a::ParamBlockArray,b::ParamBlockBroadcast)
  map(Base.materialize!,blocks(a),blocks(b))
  a
end

function Base.map(f,a::ParamBlockArray)
  map(i->f(a[i]),eachindex(a))
end
