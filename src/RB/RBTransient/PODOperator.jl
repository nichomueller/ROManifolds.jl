function ODEs.allocate_odeopcache(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.odeop,r,us)
end

function ODEs.update_odeopcache!(
  odeopcache,
  op::PODOperator,
  r::TransientParamRealization)

  update_odeopcache!(odeopcache,op.odeop,r)
end

function Algebra.allocate_residual(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.odeop,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.odeop,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  residual!(b,op.odeop,r,us,odeopcache;kwargs...)
  return Snapshots(b,r)
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  jacobian!(A,op.odeop,r,us,ws,odeopcache)
  return Snapshots(A,r)
end

function jacobian_and_residual(fesolver::ODESolver,odeop::ODEParamOperator,s::AbstractSnapshots)
  us = (get_values(s),)
  r = get_realization(s)
  odecache = allocate_odecache(fesolver,odeop,r,us)
  A,b = jacobian_and_residual(fesolver,odeop,r,us,odecache)
  return Snapshots(A,r),Snapshots(b,r)
end
