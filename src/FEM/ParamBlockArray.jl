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
Base.eltype(a::ParamBlockArray{T}) where T = T
Base.eltype(a::Type{<:ParamBlockArray{T}}) where T = T
Base.ndims(a::ParamBlockArray{T,N} where T) where N = N
Base.ndims(a::Type{<:ParamBlockArray{T,N}} where T) where N = N
Base.first(a::ParamBlockArray) = getindex(a,1)

function Base.getindex(a::ParamBlockArray,i::Integer)
  blocksi = getindex.(blocks(a),i)
  BlockArrays._BlockArray(blocksi,axes(a))
end

# iterate doesn't work properly for ParamBlockArrays...why?
# function Base.iterate(a::ParamBlockArray)
#   state = 1
#   astate = getindex(a,state)
#   astate,state
# end

# function Base.iterate(a::ParamBlockArray,state)
#   if state >= length(a)
#     return nothing
#   end
#   state += 1
#   astate = getindex(a,state)
#   astate,state
# end

#sadly, must force map function
function Base.map(f,a::ParamBlockArray)
  map(1:length(a)) do i
    f(a[i])
  end
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

function LinearAlgebra.fillstored!(a::ParamBlockMatrix,v)
  map(ai->LinearAlgebra.fillstored!(ai,v),a)
end

function Base.fill!(a::ParamBlockVector,v)
  map(ai->fill!(ai,v),a)
end

function LinearAlgebra.mul!(
  c::ParamBlockArray,
  a::ParamBlockArray,
  b::ParamBlockArray,
  α::Number,β::Number)

  @assert length(a) == length(b) == length(c)
  @inbounds for i = 1:length(a)
    mul!(c[i],a[i],b[i],α,β)
  end
  c
end

function LinearAlgebra.ldiv!(a::ParamBlockArray,m::LU,b::ParamBlockArray)
  @assert length(a) == length(b)
  @inbounds for i = 1:length(a)
    ldiv!(a[i],m,b[i])
  end
  a
end

function LinearAlgebra.ldiv!(a::ParamBlockArray,m::AbstractArray,b::ParamBlockArray)
  @assert length(a) == length(m) == length(b)
  @inbounds for i = 1:length(a)
    ldiv!(a[i],m[i],b[i])
  end
  a
end

function LinearAlgebra.rmul!(a::ParamBlockArray,b::Number)
  map(a) do a
    rmul!(a,b)
  end
end

function LinearAlgebra.lu(a::ParamBlockArray)
  map(a) do a
    lu(a)
  end
end

function LinearAlgebra.lu!(a::ParamBlockArray,b::ParamBlockArray)
  @assert length(a) == length(b)
  @inbounds for i = 1:length(a)
    lu!(a[i],b[i])
  end
  a
end
