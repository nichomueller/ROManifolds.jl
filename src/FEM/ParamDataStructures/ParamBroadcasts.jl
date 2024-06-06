abstract type AbstractParamBroadcast end

struct ParamBroadcast{D} <: AbstractParamBroadcast
  data::D
  plength::Int
end

Base.axes(A::ParamBroadcast) = (Base.OneTo(param_length(A)),)

param_length(A::ParamBroadcast) = A.plength
all_data(A::ParamBroadcast) = A.data

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast}...)
  bc = Base.broadcasted(f,map(all_data,A)...)
  plength = _find_param_length(A...)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast},b::Number)
  bc = Base.broadcasted(f,all_data(A),b)
  plength = param_length(A)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,a::Number,B::Union{AbstractParamArray,ParamBroadcast})
  bc = Base.broadcasted(f,a,all_data(B))
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

function Base.materialize(b::ParamBroadcast)
  param_materialize(Base.materialize(all_data(b)))
end

param_materialize(A) = A
param_materialize(A::AbstractArray{<:AbstractArray}) = error("what?")#ParamArray(A)
param_materialize(A::AbstractArray{<:Number}) = ArrayOfArrays(A)
param_materialize(A::SparseMatrixCSC) = MatrixOfSparseMatricesCSC(A)

function Base.materialize!(A::AbstractParamArray,b::Broadcast.Broadcasted)
  dA = all_data(A)
  Base.materialize!(dA,b)
  A
end

function Base.materialize!(A::AbstractParamArray,b::ParamBroadcast)
  dA = all_data(A)
  db = all_data(b)
  Base.materialize!(dA,db)
  A
end
