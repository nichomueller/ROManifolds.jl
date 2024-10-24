struct ParamStageOperator{T} <: ParamNonlinearOperator
  op::ODEParamOperator{T}
  cache::AbstractParamCache
  r::TransientRealization
  us::Function
  ws::Tuple{Vararg{Real}}
end

function Algebra.allocate_residual(nlop::ParamStageOperator,x::AbstractParamVector)
  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  allocate_residual(op,r,usx,cache)
end

function Algebra.residual!(
  b::AbstractParamVector,
  nlop::ParamStageOperator,
  x::AbstractParamVector)

  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  residual!(b,op,r,usx,cache)
end

function Algebra.allocate_jacobian(nlop::ParamStageOperator,x::AbstractParamVector)
  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  allocate_jacobian(op,r,usx,cache)
end

function Algebra.jacobian!(
  A::AbstractParamMatrix,
  nlop::ParamStageOperator,
  x::AbstractParamVector)

  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  ws = nlop.ws
  jacobian!(A,op,r,usx,ws,cache)
  A
end

# linear case

function Algebra.allocate_residual(
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  cache = lop.cache
  b = copy(cache.b)
  fill!(b,zero(eltype(b)))
  b
end

function Algebra.residual!(
  b::AbstractParamVector,
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  cache = lop.cache
  usx = lop.us(x)
  for k in 1:get_order(lop.op)+1
    mul!(b,cache.A[k],usx[k],1,1)
  end
  axpy!(1,cache.b,b)
  b
end

function Algebra.allocate_jacobian(
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  cache = lop.cache
  A = copy(first(cache.A))
  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  A
end

function Algebra.jacobian!(
  A::AbstractParamMatrix,
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  cache = lop.cache
  for k in 1:get_order(lop.op)+1
    axpy!(1,cache.A[k],A)
  end
  A
end
