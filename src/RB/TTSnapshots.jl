abstract type TTSnapshots{M,T} <: AbstractSnapshots{M,T} end

struct BasicTTSnapshots{M,T,P,R} <: TTSnapshots{M,T}
  mode::M
  values::P
  realization::R
  function BasicTTSnapshots(
    values::P,
    realization::R,
    mode::M=Mode1Axis(),
    ) where {M,P<:AbstractParamContainer,R}

    T = eltype(P)
    new{M,T,P,R}(mode,values,realization)
  end
end

function BasicSnapshots(values::ParamTTArray,realization::TransientParamRealization,args...)
  BasicTTSnapshots(values,realization,args...)
end

function BasicSnapshots(s::BasicTTSnapshots)
  s
end

num_space_dofs(s::BasicTTSnapshots) = length(first(s.values))

function change_mode(s::BasicTTSnapshots{Mode1Axis})
  BasicTTSnapshots(s.values,s.realization,Mode2Axis())
end

function change_mode(s::BasicTTSnapshots{Mode2Axis})
  BasicTTSnapshots(s.values,s.realization,Mode1Axis())
end

function tensor_getindex(s::BasicTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

function tensor_setindex!(s::BasicTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace] = v
end

struct TransientTTSnapshots{M,T,P,R,V} <: TTSnapshots{M,T}
  mode::M
  values::V
  realization::R
  function TransientTTSnapshots(
    values::AbstractVector{P},
    realization::R,
    mode::M=Mode1Axis(),
    ) where {M,P<:AbstractParamContainer,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    new{M,T,P,R,V}(mode,values,realization)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},realization::TransientParamRealization,args...) where P<:ParamTTArray
  TransientTTSnapshots(values,realization,args...)
end

num_space_dofs(s::TransientTTSnapshots) = length(first(first(s.values)))

function change_mode(s::TransientTTSnapshots{Mode1Axis})
  TransientTTSnapshots(s.values,s.realization,Mode2Axis())
end

function change_mode(s::TransientTTSnapshots{Mode2Axis})
  TransientTTSnapshots(s.values,s.realization,Mode1Axis())
end

function tensor_getindex(s::TransientTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace]
end

function tensor_setindex!(s::TransientTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace] = v
end

function BasicSnapshots(
  s::TransientTTSnapshots{M,T,<:ParamTTArray{T,N,A}}
  ) where {M,T,N,A}

  nt = num_times(s)
  np = num_params(s)
  array = Vector{eltype(A)}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = slow_index(i,np)
    ip = fast_index(i,np)
    array[i] = s.values[it][ip]
  end
  basic_values = ParamArray(array)
  BasicSnapshots(basic_values,s.realization,s.mode)
end

function FEM.get_values(s::TransientTTSnapshots)
  get_values(BasicSnapshots(s))
end
