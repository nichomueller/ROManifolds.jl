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

function Base.materialize(B::ParamBroadcast)
  param_materialize(map(Base.materialize,param_data(B)))
end

param_materialize(A) = A
param_materialize(A::AbstractArray{<:AbstractArray{<:Number}}) = ArrayOfArrays(A)
param_materialize(A::AbstractArray{<:SparseMatrixCSC}) = MatrixOfSparseMatricesCSC(A)
param_materialize(A::AbstractArray{<:AbstractArray{<:AbstractArray}}) = mortar(A)

function Base.materialize!(A::AbstractParamArray,b::Broadcast.Broadcasted)
  map(a -> Base.materialize!(a,b),param_data(A))
  A
end

function Base.materialize!(A::AbstractParamArray,B::ParamBroadcast)
  map(Base.materialize!,param_data(A),param_data(B))
  A
end
