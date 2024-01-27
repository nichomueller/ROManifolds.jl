struct TransientSnapshots{T,A,Np,Nt,S} <: AbstractVector{ParamVector{T,A,L}}
  snaps::S
  function TransientSnapshots(snaps::AbstractVector{ParamVector{T,A,Np}}) where {T,A,Np}
    S = typeof(snaps)
    Nt = length(snaps)
    new{T,A,Np,Nt,S}(snaps)
  end
end

const AffineTransientSnapshots{T,A,S} = TransientSnapshots{T,A,1,S}

Base.length(::TransientSnapshots{T,A,Np,Nt}) where {T,A,Np,Nt} = Np*Nt
Base.size(s::TransientSnapshots) = (num_space_dofs(s),num_times(s),num_params(s))
Base.axes(s::TransientSnapshots) = Base.OneTo.(size(s))
Base.eltype(::TransientSnapshots{T,A,Np,Nt,S}) where {T,A,Np,Nt,S} = T
Base.eltype(::Type{TransientSnapshots{T,A,Np,Nt,S}}) where {T,A,Np,Nt,S} = T
Base.ndims(::TransientSnapshots) = 1
Base.ndims(::Type{<:TransientSnapshots}) = 1
Base.first(s::TransientSnapshots) = testitem(s)
Base.getindex(s::TransientSnapshots,i...) = get_array(s)[i...]
Base.setindex!(s::TransientSnapshots,v,i...) = get_array(s)[i...] = v

function Base.show(io::IO,::MIME"text/plain",s::TransientSnapshots{T,N,A,L}) where {T,N,A,L}
  println(io, "Parametric vector of types $(eltype(A)) and length $L, with entries:")
  show(io,s.array)
end

function Base.copy(s::TransientSnapshots)
  ai = testitem(s)
  b = Vector{typeof(ai)}(undef,length(s))
  @inbounds for i = eachindex(s)
    b[i] = copy(s[i])
  end
  TransientSnapshots(b)
end

function Base.copy!(s::TransientSnapshots,b::TransientSnapshots)
  @assert length(s) == length(b)
  copyto!(s,b)
end

function Base.copyto!(s::TransientSnapshots,b::TransientSnapshots)
  map(copy!,get_array(s),get_array(b))
  s
end

function Base.similar(
  s::TransientSnapshots{T},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(s)) where {T,S}

  elb = similar(testitem(s),element_type,dims)
  b = Vector{typeof(elb)}(undef,length(s))
  @inbounds for i = eachindex(s)
    b[i] = similar(s[i],element_type,dims)
  end
  TransientSnapshots(b)
end
