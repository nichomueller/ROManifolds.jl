struct ParamBlockTTArray{T,N,A,B} <: AbstractParamContainer{T,N}
  blockarrays::A
  blockmaps::B
  function ParamBlockTTArray(
    blockarrays::A,
    blockmaps::B
    ) where {T,N,I<:AbstractIndexMap,A<:ParamBlockArray{T,N},B<:AbstractArray{I}}

    new{T,N,A,B}(blockarrays,blockmaps)
  end
end

const ParamBlockTTVector = ParamBlockTTArray{T,1,A,B} where {T,L,A<:ParamBlockVector,B}
const ParamBlockTTMatrix = ParamBlockTTArray{T,2,A,B} where {T,L,A<:ParamBlockMatrix,B}

Arrays.get_array(a::ParamBlockTTArray) = a.blockarrays
get_index_map(a::ParamBlockTTArray) = a.index_map

function _tt_mortar(ttblocks::AbstractArray{<:AbstractParamContainer})
  blocks = mortar(map(get_values,ttblocks))
  block_index_map = map(get_index_map,ttblocks)
  ParamBlockTTArray(blocks,block_index_map)
end

Base.axes(a::ParamBlockTTArray) = axes(get_array(a))
Base.size(a::ParamBlockTTArray) = size(get_array(a))
Base.length(a::ParamBlockTTArray) = length(get_array(a))
Base.length(::Type{<:ParamBlockTTArray{T,N,A}}) where {T,N,A} = length(A)
BlockArrays.blocks(a::ParamBlockTTArray) = blocks(get_array(a))

Base.getindex(a::ParamBlockTTArray,i...) = getindex(get_array(a),i...)
Base.setindex!(a::ParamBlockTTArray,v,i...) = setindex!(get_array(a),v,i...)

function Base.show(io::IO,::MIME"text/plain",a::ParamBlockTTArray{T,N,A}) where {T,N,A}
  elt = eltype(eltype(eltype(A)))
  s = size(blocks(a))
  L = length(a)
  _nice_size(s::Tuple{Int}) = "$(s[1]) - "
  _nice_size(s::Tuple{Int,Int}) = "$(s[1]) × $(s[2]) - "
  println(io, _nice_size(s) *"parametric block tt-array of types $elt and length $L, with entries:")
  show(io,a.blockarrays.blockarrays)
end

function Base.copy(a::ParamBlockTTArray)
  ParamBlockTTArray(map(copy,get_array(a)),a.blockmaps)
end

function Base.similar(a::ParamBlockTTArray)
  ParamBlockTTArray(map(similar,get_array(a)),a.blockmaps)
end

function Base.fill!(a::ParamBlockTTArray,v)
  fill!(get_array(a),v)
end

function LinearAlgebra.fillstored!(a::ParamBlockTTArray,v)
  fillstored!(get_array(a),v)
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::ParamBlockTTArray,b::ParamBlockTTArray)
      @check get_index_map(a) == get_index_map(b)
      ParamBlockTTArray(($op)(get_array(a),get_array(b)),get_index_map(a))
    end
  end
end

function Base.:*(a::ParamBlockTTArray,b::Number)
  ParamBlockTTArray(get_array(a)*b,get_index_map(a))
end

function Base.:*(a::ParamBlockTTMatrix,b::ParamBlockTTVector)
  @check get_index_map(a) == get_index_map(b)
  ParamBlockTTArray(get_array(a)*get_array(b),get_index_map(a))
end

function Base.:\(a::ParamBlockTTMatrix,b::ParamBlockTTVector)
  @check get_index_map(a) == get_index_map(b)
  ParamBlockTTArray(get_array(a)\get_array(b),get_index_map(a))
end

function Base.transpose(a::ParamBlockTTArray)
  ParamBlockTTArray(transpose(get_array(a)),get_index_map(a))
end

struct BlockTTBroadcast{V,M}
  values::V
  index_map::M
end

Arrays.get_array(a::BlockTTBroadcast) = a.values
get_index_map(a::BlockTTBroadcast) = a.index_map

function Base.broadcasted(f,a::Union{ParamBlockTTArray,BlockTTBroadcast}...)
  index_map = get_index_map(a[1])
  @check all(get_index_map.(a) == index_map)
  BlockTTBroadcast(Base.broadcasted(f,map(get_array,a)...),index_map)
end

function Base.broadcasted(f,a::Union{ParamBlockTTArray,BlockTTBroadcast},b::Number)
  BlockTTBroadcast(Base.broadcasted(f,get_array(a),b),get_index_map(a))
end

function Base.broadcasted(f,a::Number,b::Union{ParamBlockTTArray,BlockTTBroadcast})
  BlockTTBroadcast(Base.broadcasted(f,a,get_array(b)),get_index_map(a))
end

function Base.broadcasted(f,
  a::Union{ParamBlockTTArray,BlockTTBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{ParamBlockTTArray,BlockTTBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::BlockTTBroadcast)
  a = Base.materialize(get_array(b))
  mortar(a,get_index_map(a))
end

function Base.materialize!(a::ParamBlockTTArray,b::Broadcast.Broadcasted)
  Base.materialize!(get_array(a),b)
  a
end

function Base.materialize!(a::ParamBlockTTArray,b::BlockTTBroadcast)
  Base.materialize!(get_array(a),get_array(b))
  a
end
