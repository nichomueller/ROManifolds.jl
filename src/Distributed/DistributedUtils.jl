function PartitionedArrays.own_values(values::ParamArray,indices)
  o = map(a->own_values(a,indices),values)
  ParamArray(o)
end

function PartitionedArrays.ghost_values(values::ParamArray,indices)
  g = map(a->ghost_values(a,indices),values)
  ParamArray(g)
end

function PartitionedArrays.assembly_buffers(
  values::ParamArray{T,N,A,L},
  local_indices_snd,
  local_indices_rcv) where {T,N,A,L}

  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = allocate_param_array(data,L)
  buffer_snd = JaggedArray(ptdata,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = allocate_param_array(data,L)
  buffer_rcv = JaggedArray(ptdata,ptrs)
  buffer_snd,buffer_rcv
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientParamFEOperator,
  r::TransientParamRealization,
  xh::TransientDistributedCellField,
  γ::Tuple{Vararg{Real}})

  _matdata_jacobians = TransientFETools.fill_jacobians(op,μ,t,xh,γ)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end
