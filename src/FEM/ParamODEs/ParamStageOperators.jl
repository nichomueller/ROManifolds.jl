"""
    struct ParamStageOperator{O} <: NonlinearParamOperator
      op::ODEParamOperator{O}
      r::TransientRealization
      us::Function
      ws::Tuple{Vararg{Real}}
      paramcache::AbstractParamCache
    end

Stage operator to solve a parametric ODE with a time marching scheme
"""
struct ParamStageOperator{O} <: NonlinearParamOperator
  op::ODEParamOperator{O}
  r::TransientRealization
  us::Function
  ws::Tuple{Vararg{Real}}
  paramcache::AbstractParamCache
end

function ParamStageOperator(
  op::ODEParamOperator,
  r::TransientRealization,
  us::Function,
  ws::Tuple{Vararg{Real}})

  paramcache = allocate_paramcache(op,r)
  ParamStageOperator(op,r,us,ws,paramcache)
end

function Base.getindex(nlop::ParamStageOperator,timestep::Integer)
  rt = get_at_timestep(nlop.r,timestep)
  ParamStageOperator(op,rt,us,ws,paramcache)
end

initial_realization(nlop::ParamStageOperator) = get_at_time(nlop.r,:initial)
final_realization(nlop::ParamStageOperator) = get_at_time(nlop.r,:final)
ParamDataStructures.get_initial_time(nlop::ParamStageOperator) = get_initial_time(nlop.r)
ParamDataStructures.get_final_time(nlop::ParamStageOperator) = get_final_time(nlop.r)
ParamDataStructures.num_params(nlop::ParamStageOperator) = num_params(nlop.r)
ParamDataStructures.num_times(nlop::ParamStageOperator) = num_times(nlop.r)

function Algebra.allocate_residual(
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.us(x)
  allocate_residual(nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.us(x)
  residual!(b,nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.us(x)
  allocate_jacobian(nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.us(x)
  jacobian!(A,nlop.op,nlop.r,usx,nlop.ws,nlop.paramcache)
  A
end

function Algebra.solve(
  x::AbstractVector,
  ls::LinearSolver,
  op::ParamStageOperator,
  cache::Nothing)

  msg = """ Must preallocate a cache at the beginning of the time marching scheme.
  Check the Base.iterate function for a ODEParamSolution
  """
  error(msg)
end

# # linear case

# function Algebra.allocate_residual(
#   lop::ParamStageOperator{LinearParamODE},
#   x::AbstractParamVector)

#   cache = lop.cache
#   b = copy(cache.b)
#   fill!(b,zero(eltype(b)))
#   b
# end

# function Algebra.residual!(
#   b::AbstractParamVector,
#   lop::ParamStageOperator{LinearParamODE},
#   x::AbstractParamVector)

#   op = lop.op
#   cache = lop.cache.paramcache
#   r = lop.r
#   usx = lop.us(x)
#   residual!(b,op,r,usx,cache)
# end

# function Algebra.allocate_jacobian(
#   lop::ParamStageOperator{LinearParamODE},
#   x::AbstractParamVector)

#   cache = lop.cache
#   A = copy(first(cache.A))
#   A
# end

# function Algebra.jacobian!(
#   A::AbstractParamMatrix,
#   lop::ParamStageOperator{LinearParamODE},
#   x::AbstractParamVector)

#   op = lop.op
#   cache = lop.cache
#   r = lop.r
#   usx = lop.us(x)
#   ws = lop.ws
#   jacobian!(A,op,r,usx,ws,cache)
# end

# function Algebra.solve!(
#   x::AbstractParamVector,
#   ls::LinearSolver,
#   lop::ParamStageOperator{LinearParamODE},
#   cache::Nothing)

#   fill!(x,zero(eltype(x)))
#   b = residual(lop,x)
#   A = jacobian(lop,x)
#   ns = solve!(x,ls,A,b)

#   Algebra.LinearSolverCache(A,b,ns)
# end

# function Algebra.solve!(
#   x::AbstractParamVector,
#   ls::LinearSolver,
#   lop::ParamStageOperator{LinearParamODE},
#   cache::Algebra.LinearSolverCache)

#   fill!(x,zero(eltype(x)))
#   b = cache.b
#   A = cache.A
#   ns = cache.ns
#   residual!(b,lop,x)
#   jacobian!(A,lop,x)
#   ns = solve!(x,ls,A,b)
#   cache
# end
