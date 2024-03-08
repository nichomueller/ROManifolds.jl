abstract type TTSnapshots{M,T,N} <: AbstractSnapshots{M,T,N} end

#= representation of a standard tensor-train snapshot
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x1,t2,μ1) ⋯ u(x1,t2,μP)] [u(x1,t3,μ1) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] [u(xN,t2,μ1) ⋯ u(xN,t2,μP)] [u(xN,t3,μ1) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#

struct BasicTTSnapshots{M,T,N,P,R} <: TTSnapshots{M,T,N}
  mode::M
  values::P
  realization::R
  function BasicTTSnapshots(
    values::P,
    realization::R,
    mode::M=Mode1Axis(),
    ) where {M,D,P<:ParamTTArray{D},R}

    T = eltype(P)
    N = D+2
    new{M,T,N,P,R}(mode,values,realization)
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

function Base.getindex(s::BasicTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

function Base.setindex!(s::BasicTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace] = v
end

struct TransientTTSnapshots{M,T,N,P,R,V} <: TTSnapshots{M,T,N}
  mode::M
  values::V
  realization::R
  function TransientTTSnapshots(
    values::AbstractVector{P},
    realization::R,
    mode::M=Mode1Axis(),
    ) where {M,D,P<:ParamTTArray{D},R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    N = D+2
    new{M,T,N,P,R,V}(mode,values,realization)
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
