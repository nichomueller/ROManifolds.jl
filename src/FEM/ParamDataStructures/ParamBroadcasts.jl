abstract type AbstractParamBroadcast end

struct ParamBroadcast{D} <: AbstractParamBroadcast
  data::D
end

param_data(A::ParamBroadcast) = A.data

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(param_data,A)...)
  ParamBroadcast(bc)
end

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),param_data(A))
  ParamBroadcast(bc)
end

function Base.broadcasted(f,a::Number,B::Union{AbstractParamArray,ParamBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),param_data(B))
  ParamBroadcast(bc)
end

function Base.broadcasted(f,
  A::Union{AbstractParamArray,ParamBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,A,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  B::Union{AbstractParamArray,ParamBroadcast})
  Base.broadcasted(f,Base.materialize(a),B)
end

function Base.materialize(b::ParamBroadcast)
  A = map(Base.materialize,param_data(b))
  ArrayOfSimilarArrays(A)
end

function Base.materialize!(A::AbstractParamArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),param_data(A))
  A
end

function Base.materialize!(A::AbstractParamArray,b::ParamBroadcast)
  map(Base.materialize!,param_data(A),param_data(b))
  A
end
