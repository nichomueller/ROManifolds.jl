abstract type ParamStageOperator <: NonlinearOperator end

struct NonlinearParamStageOperator <: ParamStageOperator
  odeop::ODEOperator
  odeopcache
  rx::TransientParamRealization
  usx::Function
  ws::Tuple{Vararg{Real}}
end

function Algebra.allocate_residual(nlop::NonlinearStageOperator,x::AbstractVector)
  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  allocate_residual(odeop,rx,usx,odeopcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::NonlinearStageOperator,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  residual!(b,odeop,rx,usx,odeopcache)
end

function Algebra.allocate_jacobian(nlop::NonlinearStageOperator,x::AbstractVector)
  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  allocate_jacobian(odeop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::NonlinearStageOperator,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  ws = nlop.ws
  jacobian!(A,odeop,rx,usx,ws,odeopcache)
  A
end

struct LinearParamStageOperator <: ParamStageOperator
  A::AbstractMatrix
  b::AbstractVector
  reuse::Bool
end

function LinearParamStageOperator(
  odeop::ODEOperator,odeopcache,
  rx::TransientParamRealization,
  usx::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  A::AbstractMatrix,b::AbstractVector,reuse::Bool,sysslvrcache)

  residual!(b,odeop,rx,usx,odeopcache)
  if isnothing(sysslvrcache) || !reuse
    jacobian!(A,odeop,rx,usx,ws,odeopcache)
  end
  LinearParamStageOperator(A,b,reuse)
end

# NonlinearOperator interface
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
  x::AbstractVector,
  ls::LinearSolver,
  lop::LinearParamStageOperator,
  ns::Nothing)

  A = lop.A
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)

  b = lop.b
  rmul!(b,-1)

  solve!(x,ns,b)
  ns
end

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  lop::LinearParamStageOperator,
  ns)

  if !lop.reuse
    A = lop.A
    numerical_setup!(ns,A)
  end

  b = lop.b
  rmul!(b,-1)

  solve!(x,ns,b)
  ns
end
