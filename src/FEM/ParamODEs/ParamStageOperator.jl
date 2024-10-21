"""
    abstract type ParamStageOperator <: NonlinearOperator end

Parametric extension of the type [`StageOperator`](@ref) in [`Gridap`](@ref)

Subtypes:
- [`LinearParamStageOperator`](@ref)
- [`NonlinearParamStageOperator`](@ref)

"""
abstract type ParamStageOperator <: NonlinearOperator end

"""
"""
struct NonlinearParamStageOperator <: ParamStageOperator
  odeop::ODEOperator
  odeopcache
  rx::TransientRealization
  usx::Function
  ws::Tuple{Vararg{Real}}
end

function Algebra.allocate_residual(nlop::NonlinearParamStageOperator,x::AbstractVector)
  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  allocate_residual(odeop,rx,usx,odeopcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::NonlinearParamStageOperator,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  residual!(b,odeop,rx,usx,odeopcache)
end

function Algebra.allocate_jacobian(nlop::NonlinearParamStageOperator,x::AbstractVector)
  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  allocate_jacobian(odeop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::NonlinearParamStageOperator,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  ws = nlop.ws
  jacobian!(A,odeop,rx,usx,ws,odeopcache)
  A
end

"""
"""
struct LinearParamStageOperator{Ta,Tb} <: ParamStageOperator
  A::Ta
  b::Tb
end

function LinearParamStageOperator(
  odeop::ODEOperator,odeopcache,
  rx::TransientRealization,
  usx::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  matcache,veccache,sysslvrcache)

  b = residual!(veccache,odeop,rx,usx,odeopcache)
  A = jacobian!(matcache,odeop,rx,usx,ws,odeopcache)
  LinearParamStageOperator(A,b)
end

function Algebra.allocate_residual(lop::LinearParamStageOperator,x::AbstractVector)
  b = allocate_in_range(typeof(lop.b),lop.A)
  fill!(b,zero(eltype(b)))
  b
end

function Algebra.residual!(
  b::AbstractVector,
  lop::LinearParamStageOperator,
  x::AbstractVector)

  mul!(b,lop.A,x)
  axpy!(1,lop.b,b)
  b
end

function Algebra.allocate_jacobian(lop::LinearParamStageOperator,x::AbstractVector)
  lop.A
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  lop::LinearParamStageOperator,
  x::AbstractVector)

  copy_entries!(A,lop.A)
  A
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  lop::LinearParamStageOperator,
  ns::Nothing)

  A = lop.A
  A_item = testitem(A)
  ss = symbolic_setup(ls,A_item)
  ns = numerical_setup(ss,A_item)

  b = lop.b
  rmul!(b,-1)

  @inbounds for i in param_eachindex(x)
    xi = param_getindex(x,i)
    bi = param_getindex(b,i)
    solve!(xi,ns,bi)
    i == param_length(x) && continue
    Ai = param_getindex(A,i+1)
    numerical_setup!(ns,Ai)
  end

  ns
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  lop::LinearParamStageOperator,
  ns)

  A = lop.A
  b = lop.b
  rmul!(b,-1)

  @inbounds for i in param_eachindex(x)
    xi = param_getindex(x,i)
    Ai = param_getindex(A,i)
    bi = param_getindex(b,i)
    numerical_setup!(ns,Ai)
    solve!(xi,ns,bi)
  end

  ns
end

# interface for contributions

function Algebra.residual!(
  b::ArrayContribution,
  nlop::NonlinearParamStageOperator,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  residual!(b,odeop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  nlop::NonlinearParamStageOperator,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  ws = nlop.ws
  jacobian!(A,odeop,rx,usx,ws,odeopcache)
end

function Algebra.residual!(
  b::ArrayContribution,
  lop::LinearParamStageOperator,
  x::AbstractVector)

  for A in lop.A
    mul!(b,A,x)
    axpy!(1,lop.b,b)
  end
  b
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  lop::LinearParamStageOperator,
  x::AbstractVector)

  copy_entries!(A,lop.A)
  A
end
