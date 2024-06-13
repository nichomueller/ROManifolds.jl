function ODEs.allocate_odeopcache(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  odeopcache,
  op::PODOperator,
  r::TransientParamRealization)

  update_odeopcache!(odeopcache,op.op,r)
end

function Algebra.allocate_residual(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  residual!(b,op.op,r,us,odeopcache;kwargs...)
  i = get_vector_index_map(op)
  return Snapshots(b,i,r)
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  jacobian!(A,op.op,r,us,ws,odeopcache)
  i = get_matrix_index_map(op)
  return Snapshots(A,i,r)
end

function RBSteady.jacobian_and_residual(fesolver::ODESolver,odeop::ODEParamOperator,s::AbstractSnapshots)
  us = (get_values(s),)
  r = get_realization(s)
  odecache = allocate_odecache(fesolver,odeop,r,us)
  A,b = jacobian_and_residual(fesolver,odeop,r,us,odecache)
  iA = get_matrix_index_map(op)
  ib = get_vector_index_map(op)
  return Snapshots(A,iA,r),Snapshots(b,ib,r)
end
