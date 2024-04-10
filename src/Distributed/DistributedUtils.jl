function PartitionedArrays.own_values(values::ParamArray,indices)
  o = map(a->own_values(a,indices),values)
  ParamArray(o)
end

function PartitionedArrays.ghost_values(values::ParamArray,indices)
  g = map(a->ghost_values(a,indices),values)
  ParamArray(g)
end

function PartitionedArrays.assembly_buffers(
  values::ParamArray{T,N,L,A},
  local_indices_snd,
  local_indices_rcv) where {T,N,L,A}

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

const ParamVectorLocalView = GridapDistributed.LocalView{T,1,A,B} where {T,A<:ParamVector,B}
const ParamMatrixLocalView = GridapDistributed.LocalView{T,2,A,B} where {T,A<:ParamMatrix,B}
const ParamArrayLocalView = Union{ParamVectorLocalView,ParamMatrixLocalView}

function Base.getindex(a::ParamArrayLocalView,index::Integer)
  GridapDistributed.LocalView(a.plids_to_value[index],a.d_to_lid_to_plid)
end

@inline function Algebra.add_entry!(combine::Function,A::ParamMatrixLocalView,v::Number,i,j)
  for k = eachindex(A)
    aij = A[k][i,j]
    A[k][i,j] = combine(aij,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ParamVectorLocalView,v::Number,i)
  for k = eachindex(A)
    ai = A[k][i]
    A[k][i] = combine(ai,v)
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::ParamMatrixLocalView,vs::AbstractParamContainer,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          for k = eachindex(vs)
            vij = vs[k][li,lj]
            add_entry!(combine,A[k],vij,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::ParamVectorLocalView,vs::AbstractParamContainer,is)
  for (li,i) in enumerate(is)
    if i>0
      for k = eachindex(vs)
        vi = vs[k][li]
        add_entry!(combine,A[k],vi,i)
      end
    end
  end
  A
end

function Base.materialize(
  b::PartitionedArrays.PBroadcasted{<:AbstractArray{<:FEM.ParamBroadcast}})
  own_values_out = map(Base.materialize,b.own_values)
  PT = eltype(own_values_out)
  a = PVector{PT}(undef,b.index_partition)
  Base.materialize!(a,b)
  a
end
