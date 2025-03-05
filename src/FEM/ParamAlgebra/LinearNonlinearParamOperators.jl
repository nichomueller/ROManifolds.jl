"""
    get_linear_operator(op::NonlinearParamOperator) -> NonlinearParamOperator

Returns the linear part of the operator `op`
"""
get_linear_operator(op::NonlinearParamOperator) = @abstractmethod

"""
    get_nonlinear_operator(op::NonlinearParamOperator) -> NonlinearParamOperator

Returns the nonlinear part of the operator `op`
"""
get_nonlinear_operator(op::NonlinearParamOperator) = @abstractmethod

"""
    get_linear_systemcache(op::NonlinearParamOperator) -> AbstractParamCache

Returns the cache associated to the linear part of the operator `op`
"""
get_linear_systemcache(op::NonlinearParamOperator) = @abstractmethod

"""
    struct LinNonlinParamOperator <: NonlinearParamOperator
      op_linear::NonlinearParamOperator
      op_nonlinear::NonlinearParamOperator
      cache_linear::AbstractParamCache
    end

Interface that allows to split the linear part of a parametric, nonlinear, differential
operator from the nonlinear part.
Fields:
- `op_linear`: linear part of the operator
- `op_nonlinear`: nonlinear part of the operator
- `cache_linear`: cache related to the linear part of the operator
"""
struct LinNonlinParamOperator <: NonlinearParamOperator
  op_linear::NonlinearParamOperator
  op_nonlinear::NonlinearParamOperator
  cache_linear::AbstractParamCache
end

get_linear_operator(op::LinNonlinParamOperator) = op.op_linear
get_nonlinear_operator(op::LinNonlinParamOperator) = op.op_nonlinear
get_linear_systemcache(op::LinNonlinParamOperator) = op.cache_linear

function allocate_paramcache(op::LinNonlinParamOperator,μ::Realization)
  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,μ)
end

function allocate_systemcache(op::LinNonlinParamOperator,u::AbstractVector)
  op_nlin = get_nonlinear_operator(op)
  allocate_systemcache(op_nlin,u)
end

function update_paramcache!(
  paramcache,
  op::LinNonlinParamOperator,
  μ::Realization)

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function Algebra.allocate_residual(
  op::LinNonlinParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  op_lin = get_linear_operator(op)
  syscache_lin = get_linear_systemcache(op)
  b_lin = get_vector(syscache_lin)
  residual!(b_lin,op_lin,u,paramcache)
  copy(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinNonlinParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  op_lin = get_linear_operator(op)
  syscache_lin = get_linear_systemcache(op)
  A_lin = get_matrix(syscache_lin)
  jacobian!(A_lin,op_lin,u,paramcache)
  copy(A_lin)
end

function Algebra.residual!(
  b,
  op::LinNonlinParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  syscache_lin = get_linear_systemcache(op)
  A_lin = get_matrix(syscache_lin)
  b_lin = get_vector(syscache_lin)

  op_nlin = get_nonlinear_operator(op)
  residual!(b,op_nlin,μ,u,paramcache)
  mul!(b,A_lin,u,1,1)
  axpy!(1,b_lin,b)

  b
end

function ODEs.jacobian_add!(
  A,
  op::LinNonlinParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  syscache_lin = get_linear_systemcache(op)
  A_lin = get_matrix(syscache_lin)

  op_nlin = get_nonlinear_operator(op)
  jacobian_add!(A,op_nlin,μ,u,paramcache)
  axpy!(1,A_lin,A)

  A
end

function allocate_systemcache(nlop::LinNonlinParamOperator)
  cache_linear = get_linear_systemcache(op)
  copy(cache_linear)
end
