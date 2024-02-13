# for op in (:+,:-)
#   @eval begin
#     function ($op)(s::AbstractTransientSnapshots)
#       values = map($op,get_values(s))
#       realization = get_realization(s)
#       mode = get_mode(s)
#       Snapshots(values,realization,mode)
#     end

#     function ($op)(s::CompressedTransientSnapshots)
#       values = ($op)(get_values(s))
#       realization = get_realization(s)
#       cur_mode = s.current_mode
#       init_mode = s.initial_mode
#       CompressedTransientSnapshots(cur_mode,init_mode,values,realization)
#     end

#     function ($op)(s::SelectedSnapshotsAtIndices)
#       SelectedSnapshotsAtIndices(($op)(s.snaps),s.selected_indices)
#     end

#     function ($op)(s::InnerTimeOuterParamTransientSnapshots)
#       values = map($op,get_values(s))
#       realization = get_realization(s)
#       InnerTimeOuterParamTransientSnapshots(values,realization)
#     end

#     function ($op)(s::SelectedInnerTimeOuterParamTransientSnapshots)
#       SelectedInnerTimeOuterParamTransientSnapshots(($op)(s.snaps),s.selected_indices)
#     end

#     # function ($op)(s::T,t::T) where T<:AbstractTransientSnapshots
#     #   values = map($op,get_values(s),get_values(t))
#     #   realization = get_realization(s)
#     #   mode = get_mode(s)
#     #   Snapshots(values,realization,mode)
#     # end

#     # function ($op)(s::CompressedTransientSnapshots)
#     #   values = ($op)(get_values(s))
#     #   realization = get_realization(s)
#     #   cur_mode = s.current_mode
#     #   init_mode = s.initial_mode
#     #   CompressedTransientSnapshots(cur_mode,init_mode,values,realization)
#     # end

#     # function ($op)(s::SelectedSnapshotsAtIndices)
#     #   SelectedSnapshotsAtIndices(($op)(s.snaps),s.selected_indices)
#     # end

#     # function ($op)(s::InnerTimeOuterParamTransientSnapshots)
#     #   values = map($op,get_values(s))
#     #   realization = get_realization(s)
#     #   InnerTimeOuterParamTransientSnapshots(values,realization)
#     # end

#     # function ($op)(s::SelectedInnerTimeOuterParamTransientSnapshots)
#     #   SelectedInnerTimeOuterParamTransientSnapshots(($op)(s.snaps),s.selected_indices)
#     # end
#   end
# end

# TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(ss'),eltype(ss))
# YE = similar(ss,TS,(size(ss',1),size(ss,2)))
function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::AbstractMatrix,
  b::A) where A<:BasicSnapshots

  np = num_params(b)
  for i = axes(a,1)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a[i,:],b.values[jp+(jt-1)*np])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:BasicSnapshots,B<:BasicSnapshots,T}

  np = num_params(b)
  for i = axes(a,1)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.parent.values[ip+(it-1)*np],b.values[jp+(jt-1)*np])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:CompressedTransientSnapshots,B<:BasicSnapshots,T}

  nt = num_params(b)
  np = num_params(b)
  for i = axes(a,1)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    icolumn = column_index(it,ip,nt,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.parent.values[:,icolumn],b.values[jp+(jt-1)*np])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::AbstractMatrix,
  b::A) where A<:TransientSnapshots

  np = num_params(b)
  for i = axes(a,1)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a[i,:],b.values[jt][jp])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:TransientSnapshots,B<:TransientSnapshots,T}

  np = num_params(b)
  for i = axes(a,1)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.parent.values[it][ip],b.values[jt][jp])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:CompressedTransientSnapshots,B<:TransientSnapshots,T}

  nt = num_params(b)
  np = num_params(b)
  for i = axes(a,1)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    icolumn = column_index(it,ip,nt,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.parent.values[:,icolumn],b.values[jt][jp])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::AbstractMatrix,
  b::A) where A<:SelectedSnapshotsAtIndices

  Js = space_indices(b)
  Jt = time_indices(b)
  Jp = param_indices(b)
  np = num_params(b)
  for i = axes(a,1)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a[i,:],b.values[Jt[jt]][Jp[jp]][Js])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::A,
  b::B) where {A<:SelectedSnapshotsAtIndices,B<:SelectedSnapshotsAtIndices}

  Is = space_indices(a)
  It = time_indices(a)
  Ip = param_indices(a)
  Js = space_indices(b)
  Jt = time_indices(b)
  Jp = param_indices(b)
  np = num_params(b)
  for i = axes(a,2)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.values[It[it]][Ip[ip]][Is],b.values[Jt[jt]][Jp[jp]][Js])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:SelectedSnapshotsAtIndices,B<:SelectedSnapshotsAtIndices,T}

  Is = space_indices(a.parent)
  It = time_indices(a.parent)
  Ip = param_indices(a.parent)
  Js = space_indices(b)
  Jt = time_indices(b)
  Jp = param_indices(b)
  np = num_params(b)
  for i = axes(a,1)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.parent.snaps.values[It[it]][Ip[ip]][Is],b.snaps.values[Jt[jt]][Jp[jp]][Js])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::AbstractMatrix,
  b::A) where A<:InnerTimeOuterParamTransientSnapshots

  nt = num_times(b)
  for i = axes(a,1)
    for j = axes(b,2)
      jt = fast_index(j,nt)
      jp = slow_index(j,nt)
      @inbounds c[i,j] = dot(a[i,:],b.values[jp][jt])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:InnerTimeOuterParamTransientSnapshots,B<:InnerTimeOuterParamTransientSnapshots,T}

  nt = num_times(b)
  for i = axes(a,1)
    it = fast_index(i,nt)
    ip = slow_index(i,nt)
    for j = axes(b,2)
      jt = fast_index(j,nt)
      jp = slow_index(j,nt)
      @inbounds c[i,j] = dot(a.parent.values[ip][it],b.values[jp][jt])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:CompressedTransientSnapshots,B<:InnerTimeOuterParamTransientSnapshots,T}

  nt = num_params(b)
  np = num_params(b)
  for i = axes(a,1)
    it = fast_index(i,nt)
    ip = slow_index(i,nt)
    icolumn = column_index(it,ip,nt,np)
    for j = axes(b,2)
      jt = fast_index(j,nt)
      jp = slow_index(j,nt)
      @inbounds c[i,j] = dot(a.parent.values[:,icolumn],b.values[jp][jt])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::AbstractMatrix,
  b::A) where A<:SelectedInnerTimeOuterParamTransientSnapshots

  Js = space_indices(b)
  Jt = time_indices(b)
  Jp = param_indices(b)
  np = num_params(b)
  for i = axes(a,1)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a[i,:],b.values[Jp[jp]][Jt[jt]][Js])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::A,
  b::B) where {A<:SelectedInnerTimeOuterParamTransientSnapshots,B<:SelectedInnerTimeOuterParamTransientSnapshots}

  Is = space_indices(a)
  It = time_indices(a)
  Ip = param_indices(a)
  Js = space_indices(b)
  Jt = time_indices(b)
  Jp = param_indices(b)
  np = num_params(b)
  for i = axes(a,2)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.values[Ip[ip]][It[it]][Is],b.values[Jp[jp]][Jt[jt]][Js])
    end
  end
  return c
end

function LinearAlgebra.mul!(
  c::AbstractMatrix,
  a::Adjoint{T,A},
  b::B) where {A<:SelectedInnerTimeOuterParamTransientSnapshots,B<:SelectedInnerTimeOuterParamTransientSnapshots,T}

  Is = space_indices(a.parent)
  It = time_indices(a.parent)
  Ip = param_indices(a.parent)
  Js = space_indices(b)
  Jt = time_indices(b)
  Jp = param_indices(b)
  np = num_params(b)
  for i = axes(a,1)
    it = slow_index(i,np)
    ip = fast_index(i,np)
    for j = axes(b,2)
      jt = slow_index(j,np)
      jp = fast_index(j,np)
      @inbounds c[i,j] = dot(a.parent.values[Ip[ip]][It[it]][Is],b.values[Jp[jp]][Jt[jt]][Js])
    end
  end
  return c
end
