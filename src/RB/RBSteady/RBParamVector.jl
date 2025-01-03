"""
    struct RBParamVector{T,A<:ParamVector{T},B} <: ParamArray{T,1}
      data::A
      fe_data::B
    end

Parametric vector obtained by applying a `Projection` on a high-dimensional
parametric FE vector `fe_data`, which is stored (but mostly unused) for conveniency
"""
struct RBParamVector{T,A<:ParamVector{T},B} <: ParamArray{T,1}
  data::A
  fe_data::B
end

Base.size(a::RBParamVector) = size(a.data)
Base.getindex(a::RBParamVector,i::Integer) = getindex(a.data,i)
Base.setindex!(a::RBParamVector,v,i::Integer) = setindex!(a.data,v,i)
ParamDataStructures.param_length(a::RBParamVector) = param_length(a.data)
ParamDataStructures.get_all_data(a::RBParamVector) = get_all_data(a.data)
ParamDataStructures.param_getindex(a::RBParamVector,i::Integer) = param_getindex(a.data,i)

function Base.copy(a::RBParamVector)
  data′ = copy(a.data)
  fe_data′ = copy(a.fe_data)
  RBParamVector(data′,fe_data′)
end

function Base.similar(A::RBParamVector{T},::Type{S}) where {T,S<:AbstractVector}
  data′ = similar(a.data,S)
  fe_data′ = copy(a.fe_data)
  RBParamVector(data′,fe_data′)
end

function Base.similar(A::RBParamVector{T},::Type{S},dims::Dims{1}) where {T,S<:AbstractVector}
  data′ = similar(a.data,S,dims)
  fe_data′ = similar(a.fe_data,S,dims)
  RBParamVector(data′,fe_data′)
end

function Base.copyto!(a::RBParamVector,b::RBParamVector)
  copyto!(a.data,b.data)
  copyto!(a.fe_data,b.fe_data)
  a
end

function Base.fill!(a::RBParamVector,b::Number)
  fill!(a.data,b)
  return a
end

function project(r::RBSpace,a::RBParamVector)
  project!(a.data,r,a.fe_data)
end

function inv_project(r::RBSpace,a::RBParamVector)
  inv_project!(a.fe_data,r,a.data)
end
