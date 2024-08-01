"""
    abstract type AbstractParamBroadcast end

Represents a broadcast operation over parametric quantities. This allows to perform
operations that are usually not allowed in a normal broadcasting setting, e.g.
adding a scalar to an array of arrays. Subtypes:

- [`ParamBroadcast`]

"""
abstract type AbstractParamBroadcast end

struct ParamBroadcast{D} <: AbstractParamBroadcast
  data::D
  plength::Int
end

Base.axes(A::ParamBroadcast) = (Base.OneTo(param_length(A)),)

param_data(A::ParamBroadcast) = A.data
param_length(A::ParamBroadcast) = A.plength

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(param_data,A)...)
  plength = find_param_length(A...)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),param_data(A))
  plength = param_length(A)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,a::Number,B::Union{AbstractParamArray,ParamBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),param_data(B))
  plength = param_length(B)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,A::Union{AbstractParamVector,ParamBroadcast},b::AbstractVector{<:Number})
  @check length(b) == param_length(A)
  bc = map((a,b)->Base.broadcasted(f,a,b),param_data(A),b)
  plength = param_length(A)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,a::AbstractVector{<:Number},B::Union{AbstractParamVector,ParamBroadcast})
  @check length(a) == param_length(B)
  bc = map((a,b)->Base.broadcasted(f,a,b),a,param_data(B))
  plength = param_length(B)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,
  A::Union{AbstractParamArray,ParamBroadcast},
  b::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle})
  Base.broadcasted(f,A,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle},
  B::Union{AbstractParamArray,ParamBroadcast})
  Base.broadcasted(f,Base.materialize(a),B)
end

function Base.materialize(B::ParamBroadcast)
  param_materialize(map(Base.materialize,param_data(B)))
end

param_materialize(A) = A
param_materialize(A::AbstractArray{<:AbstractArray{<:Number}}) = ArrayOfArrays(A)
param_materialize(A::AbstractArray{<:SparseMatrixCSC}) = MatrixOfSparseMatricesCSC(A)

function Base.materialize!(A::AbstractParamArray,b::Broadcast.Broadcasted)
  map(a -> Base.materialize!(a,b),param_data(A))
  A
end

function Base.materialize!(A::AbstractParamArray,B::ParamBroadcast)
  map(Base.materialize!,param_data(A),param_data(B))
  A
end

struct ConsecutiveParamBroadcast{D} <: AbstractParamBroadcast
  data::D
  plength::Int
end

# Base.axes(A::ConsecutiveParamBroadcast) = (Base.OneTo(param_length(A)),)

consecutive_data(A::ConsecutiveArrayOfArrays) = A.data
consecutive_data(A::ConsecutiveParamBroadcast) = A.data
param_length(A::ConsecutiveParamBroadcast) = A.plength

function Base.broadcasted(f,A::Union{ConsecutiveArrayOfArrays,ConsecutiveParamBroadcast}...)
  bc = Base.broadcasted(f,map(consecutive_data,A)...)
  plength = find_param_length(A...)
  ConsecutiveParamBroadcast(bc,plength)
end

function Base.broadcasted(f,A::Union{ConsecutiveArrayOfArrays,ConsecutiveParamBroadcast},b::Number)
  bc = Base.broadcasted(f,consecutive_data(A),b)
  plength = param_length(A)
  ConsecutiveParamBroadcast(bc,plength)
end

function Base.broadcasted(f,a::Number,B::Union{ConsecutiveArrayOfArrays,ConsecutiveParamBroadcast})
  bc = Base.broadcasted(f,a,consecutive_data(B))
  plength = param_length(B)
  ConsecutiveParamBroadcast(bc,plength)
end

function Base.broadcasted(f,
  A::Union{ConsecutiveArrayOfArrays,ConsecutiveParamBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,A,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  B::Union{ConsecutiveArrayOfArrays,ConsecutiveParamBroadcast})
  Base.broadcasted(f,Base.materialize(a),B)
end

function Base.materialize(B::ConsecutiveParamBroadcast)
  ConsecutiveArrayOfArrays(Base.materialize(consecutive_data(B)))
end

function Base.materialize!(A::ConsecutiveArrayOfArrays,B::ConsecutiveParamBroadcast)
  Base.materialize!(consecutive_data(A),consecutive_data(B))
  A
end

struct TrivialConsecutiveBroadcast{D<:ConsecutiveArrayOfArrays} <: AbstractParamBroadcast
  data::D
end

function Base.broadcasted(f,A::ConsecutiveArrayOfArrays,b::AbstractVector{<:Number})
  @check length(b) == param_length(A)
  C = similar(A)
  for (i,ai) in enumerate(param_data(A))
    C[i] = f(ai,b[i])
  end
  return TrivialConsecutiveBroadcast(C)
end

function Base.broadcasted(f,a::AbstractVector{<:Number},B::ConsecutiveArrayOfArrays)
  @check length(a) == param_length(B)
  C = similar(B)
  for (i,bi) in enumerate(param_data(B))
    C[i] = f(a[i],bi)
  end
  return TrivialConsecutiveBroadcast(C)
end

function Base.materialize!(A::ConsecutiveArrayOfArrays,B::TrivialConsecutiveBroadcast)
  Base.copyto!(A,B.data)
  A
end

struct BlockParamBroadcast{A<:AbstractParamBroadcast,N} <: AbstractArray{A,N}
  data::Array{A,N}
end

Base.size(A::BlockParamBroadcast) = size(A.data)
Base.getindex(A::BlockParamBroadcast{B,N},i::Vararg{Integer,N}) where {B,N} = getindex(A.data,i...)

param_length(A::BlockParamBroadcast) = testitem(A.data)
BlockArrays.blocks(A::BlockParamBroadcast) = A.data

function Base.broadcasted(f,A::Union{BlockArrayOfArrays,BlockParamBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(blocks,A)...)
  BlockParamBroadcast(bc)
end

function Base.broadcasted(f,A::Union{BlockArrayOfArrays,BlockParamBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),blocks(A))
  BlockParamBroadcast(bc)
end

function Base.broadcasted(f,a::Number,B::Union{BlockArrayOfArrays,BlockParamBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),blocks(B))
  BlockParamBroadcast(bc)
end

function Base.broadcasted(f,
  A::Union{BlockArrayOfArrays,BlockParamBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,A,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  B::Union{BlockArrayOfArrays,BlockParamBroadcast})
  Base.broadcasted(f,Base.materialize(a),B)
end

function Base.materialize(B::BlockParamBroadcast)
  mortar(map(Base.materialize,blocks(B)))
end

function Base.materialize!(A::BlockArrayOfArrays,b::Broadcast.Broadcasted)
  map(a -> Base.materialize!(a,b),blocks(A))
  A
end

function Base.materialize!(A::BlockArrayOfArrays,B::BlockParamBroadcast)
  map(Base.materialize!,blocks(A),blocks(B))
  A
end
