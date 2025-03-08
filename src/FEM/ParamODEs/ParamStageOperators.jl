struct ShiftRules <: Function
  first_shift!::Function
  second_shift!::Function
  state_update::Function
end

function ShiftRules(solver::ODESolver,state_update::Function)
  @notimplemented "For now, only theta methods are implemented"
end

function ShiftRules(solver::ThetaMethod,state_update::Function)
  dt,θ = solver.dt,solver.θ
  first_shift!(r) = shift!(r,θ*dt)
  second_shift!(r) = shift!(r,dt*(1-θ))
  ShiftRules(first_shift!,second_shift!,state_update)
end

first_shift!(srule::ShiftRules,r::TransientRealization) = srule.first_shift!(r)
second_shift!(srule::ShiftRules,r::TransientRealization) = srule.second_shift!(r)
state_update(srule::ShiftRules,x::AbstractVector) = srule.state_update(x)

"""
    struct ParamStageOperator{O} <: NonlinearParamOperator
      op::ODEParamOperator{O}
      r::TransientRealization
      shift::ShiftRules
      ws::Tuple{Vararg{Real}}
      paramcache::AbstractParamCache
    end

Stage operator to solve a parametric ODE with a time marching scheme
"""
struct ParamStageOperator{O} <: NonlinearParamOperator
  op::ODEParamOperator{O}
  r::TransientRealization
  shift::ShiftRules
  ws::Tuple{Vararg{Real}}
  paramcache::AbstractParamCache
end

function ParamStageOperator(
  op::ODEParamOperator,
  r::TransientRealization,
  shift::ShiftRules,
  ws::Tuple{Vararg{Real}})

  r0 = get_at_time(r,:initial)
  paramcache = allocate_paramcache(op,r0)
  ParamStageOperator(op,r,shift,ws,paramcache)
end

ParamDataStructures.num_params(nlop::ParamStageOperator) = num_params(nlop.r)
ParamDataStructures.num_times(nlop::ParamStageOperator) = num_times(nlop.r)
first_shift!(nlop::ParamStageOperator,r::TransientRealization) = first_shift!(nlop.shift,r)
second_shift!(nlop::ParamStageOperator,r::TransientRealization) = second_shift!(nlop.shift,r)
state_update(nlop::ParamStageOperator,x::AbstractVector) = state_update(nlop.shift,x)

function Base.iterate(nlop::ParamStageOperator)
  timestep = 1
  ri = get_at_time(nlop.r,:initial)
  if timestep > num_times(nlop.r)
    return nothing
  end
  first_shift!(nlop,ri)
  paramcachei = update_paramcache!(nlop.paramcache,nlop.op,ri)
  nlopi = ParamStageOperator(nlop.op,ri,nlop.shift,nlop.ws,paramcachei)
  timestep += 1
  state = (timestep,ri)
  return nlopi,state
end

function Base.iterate(nlop::ParamStageOperator,state)
  timestep,ri = state
  second_shift!(nlop,ri)
  if timestep > num_times(nlop.r)
    return nothing
  end
  first_shift!(nlop,ri)
  paramcachei = update_paramcache!(nlop.paramcache,nlop.op,ri)
  nlopi = ParamStageOperator(nlop.op,ri,nlop.shift,nlop.ws,paramcachei)
  timestep += 1
  state = (timestep,ri)
  return nlopi,state
end

function Algebra.allocate_residual(
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = state_update(nlop,x)
  allocate_residual(nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = state_update(nlop,x)
  residual!(b,nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = state_update(nlop,x)
  allocate_jacobian(nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = state_update(nlop,x)
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
