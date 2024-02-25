struct ParamBlockArray{T,N,A,L,B,BS} <: AbstractParamContainer{T,N}
  blockarrays::B
  axes::BS
  function ParamBlockArray(
    blockarrays::AbstractArray{ParamArray{T,N,A,L}},
    axes::BS
    ) where {T,N,A,L,BS}
    B = typeof(blockarrays)
    new{T,N,A,L,B,BS}(blockarrays,axes)
  end
end

const ParamBlockVector{T,A,L,B} = ParamBlockArray{T,1,A,L,B}
const ParamBlockMatrix{T,A,L,B} = ParamBlockArray{T,2,A,L,B}

function BlockArrays.mortar(blocks::AbstractArray{<:AbstractParamContainer})
  block_sizes = BlockArrays.sizes_from_blocks(first.(blocks))
  block_axes = map(blockedrange,block_sizes)
  ParamBlockArray(blocks,block_axes)
end

Arrays.get_array(a::ParamBlockArray) = a.blockarrays
BlockArrays.blocks(a::ParamBlockArray) = get_array(a)
BlockArrays.blocksize(a::ParamBlockArray,i...) = size(get_array(a),i...)
BlockArrays.blocklength(a::ParamBlockArray) = length(get_array(a))
BlockArrays.eachblock(a::ParamBlockArray) = Base.OneTo(blocklength(a))
Base.length(a::ParamBlockArray{T,N,A,L}) where {T,N,A,L} = L
Base.size(a::ParamBlockArray) = map(length,axes(a))
Base.axes(a::ParamBlockArray) = a.axes
Base.eltype(::ParamBlockArray{T}) where T = T
Base.eltype(::Type{<:ParamBlockArray{T}}) where T = T
Base.ndims(::ParamBlockArray{T,N} where T) where N = N
Base.ndims(::Type{<:ParamBlockArray{T,N}} where T) where N = N
Base.eachindex(::ParamBlockArray{T,N,A,L}) where {T,N,A,L} = Base.OneTo(L)

Arrays.testitem(a::ParamBlockArray) = first(a)
Base.first(a::ParamBlockArray) = getindex(a,1)
Base.getindex(a::ParamBlockArray,i...) = ParamBlockArrayView(a,i...)
Base.getindex(a::ParamBlockArray{T},nb::Block{1}) where T = get_array(a)[nb.n...]
Base.getindex(a::ParamBlockArray{T},nb::Block{N}) where {T,N} = get_array(a)[nb.n...]

function Base.setindex!(a::ParamBlockArray,v::BlockArray,i...)
  blocksi = getindex(a,i...)
  setindex!(blocksi,v,i)
  a
end

function Base.show(io::IO,::MIME"text/plain",a::ParamBlockArray{T,N,A,L}) where {T,N,A,L}
  s = size(blocks(a))
  _nice_size(s::Tuple{Int}) = "$(s[1]) - "
  _nice_size(s::Tuple{Int,Int}) = "$(s[1]) × $(s[2]) - "
  println(io, _nice_size(s) *"parametric block array of types $(eltype(A)) and length $L, with entries:")
  show(io,a.blockarrays)
end

function Base.copy(a::ParamBlockArray)
  ParamBlockArray(copy.(get_array(a)),a.axes)
end

function Base.similar(
  a::ParamBlockArray{T},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(first(blocks(a)))) where {T,S}

  ParamBlockArray(similar.(get_array(a),element_type,dims),a.axes)
end

function LinearAlgebra.fillstored!(a::ParamBlockMatrix,v)
  for i = eachblock(a)
    LinearAlgebra.fillstored!(a.blockarrays[i],v)
  end
  a
end

function Base.fill!(a::ParamBlockVector,v)
  for i = eachblock(a)
    fill!(a.blockarrays[i],v)
  end
  a
end

function Base.:+(a::T,b::T) where T<:ParamBlockArray
  @assert a.axes == b.axes
  c = map(get_array(a),get_array(b)) do a,b
    a+b
  end
  ParamBlockArray(c,a.axes)
end

function Base.:-(a::T,b::T) where T<:ParamBlockArray
  @assert a.axes == b.axes
  c = map(get_array(a),get_array(b)) do a,b
    a-b
  end
  ParamBlockArray(c,a.axes)
end

function Base.:*(a::ParamBlockArray,b::Number)
  c = map(get_array(a)) do a
    a*b
  end
  ParamBlockArray(c,a.axes)
end

function Base.:*(a::Number,b::ParamBlockArray)
  b*a
end

function LinearAlgebra.mul!(
  c::ParamBlockArray,
  a::ParamBlockArray,
  b::ParamBlockArray,
  α::Number,β::Number)

  @assert length(a) == length(b) == length(c)
  @inbounds for i = eachindex(a)
    mul!(c[i],a[i],b[i],α,β)
  end
  c
end

function LinearAlgebra.ldiv!(a::ParamBlockArray,m::LU,b::ParamBlockArray)
  @assert length(a) == length(b)
  @inbounds for i = eachindex(a)
    ldiv!(a[i],m,b[i])
  end
  a
end

function LinearAlgebra.ldiv!(a::ParamBlockArray,m::AbstractArray,b::ParamBlockArray)
  @assert length(a) == length(m) == length(b)
  @inbounds for i = eachindex(a)
    ldiv!(a[i],m[i],b[i])
  end
  a
end

function LinearAlgebra.rmul!(a::ParamBlockArray,b::Number)
  map(get_array(a)) do a
    rmul!(a,b)
  end
end

function LinearAlgebra.lu(a::ParamBlockArray)
  lua = [lu(a[i]) for i = eachindex(a)]
  ParamContainer(lua)
end

function LinearAlgebra.lu!(a::ParamBlockArray,b::ParamBlockArray)
  @assert length(a) == length(b)
  @inbounds for i = eachindex(a)
    lu!(a[i],b[i])
  end
  a
end

struct ParamBlockBroadcast{D} <: AbstractParamBroadcast
  array::D
end

_get_array(a::ParamBlockBroadcast) = _get_array(a.array)
_get_array(a::ParamBlockArray) = get_array(a)
_get_array(a) = a

function Base.broadcasted(f,a::Union{ParamBlockArray,ParamBlockBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(_get_array,a)...)
  ParamBlockBroadcast(bc)
end

function Base.broadcasted(f,a::Union{ParamBlockArray,ParamBlockBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),_get_array(a))
  ParamBlockBroadcast(bc)
end

function Base.broadcasted(f,a::Number,b::Union{ParamBlockArray,ParamBlockBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),_get_array(b))
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
  a = map(Base.materialize,_get_array(b))
  mortar(a)
end

function Base.materialize!(a::ParamBlockArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),_get_array(a))
  a
end

function Base.materialize!(a::ParamBlockArray,b::ParamBlockBroadcast)
  map(i->Base.materialize!(_get_array(a)[i],_get_array(b)[i]),eachblock(a))
  a
end

function Base.map(f,a::ParamBlockArray)
  map(i->f(get_array(a)[i]),eachblock(a))
end

function mymortar(a::ParamBlockArray,ind::Integer)
  BlockArrays._BlockArray([block[ind] for block in get_array(a)],axes(a))
end

struct ParamBlockArrayView{T,N,A,I} <: AbstractParamContainer{T,N}
  param_blocks::A
  inds::I
  function ParamBlockArrayView(
    param_blocks::A,
    inds::I
    ) where {T,N,A<:ParamBlockArray{T,N},I<:AbstractVector}
    new{T,N,A,I}(param_blocks,inds)
  end
end

function ParamBlockArrayView(a::ParamBlockArray,ind::Integer)
  mymortar(a,ind)
end

const ParamBlockVectorView{T,A,I} = ParamBlockArrayView{T,1,A,I}
const ParamBlockMatrixView{T,A,I} = ParamBlockArrayView{T,2,A,I}

Base.eltype(a::ParamBlockArrayView) = eltype(a.param_blocks)
Base.ndims(a::ParamBlockArrayView) = ndims(a.param_blocks)
Base.size(a::ParamBlockArrayView) = size(a.param_blocks)
Base.axes(a::ParamBlockArrayView) = axes(a.param_blocks)
Base.length(a::ParamBlockArrayView) = length(a.inds)

function Base.getindex(a::ParamBlockArrayView,i...)
  blockarrays = get_array(a.param_blocks)
  ind = map(j -> getindex(a.inds,j),i)
  blocks = [block[ind] for block in blockarrays]
  ParamBlockArray(blocks,axes(a))
end

function Base.getindex(a::ParamBlockArrayView,i::Integer)
  mymortar(a.param_blocks,a.inds[i])
end

function Base.getindex(a::ParamBlockArrayView{T},nb::Block{1}) where T
  blockarrays = get_array(a.param_blocks)
  block = blockarrays[nb.n...]
  block[a.inds]
end

function Base.getindex(a::ParamBlockArrayView{T},nb::Block{N}) where {T,N}
  blockarrays = get_array(a.param_blocks)
  block = blockarrays[nb.n...]
  block[a.inds]
end
