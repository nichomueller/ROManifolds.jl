struct Snapshots{T<:AbstractArray}
  snaps::Vector{NonaffinePTArray{T}}
end

Base.length(s::Snapshots) = length(s.snaps)
Base.size(s::Snapshots,args...) = size(testitem(first(s.snaps)),args...)
Base.eachindex(s::Snapshots) = eachindex(s.snaps)
Base.lastindex(s::Snapshots) = num_params(s)
Base.copy(s::Snapshots) = Snapshots(copy.(s.snaps))
num_space_dofs(s::Snapshots) = size(s,1)
num_time_dofs(s::Snapshots) = length(s)
num_params(s::Snapshots) = length(first(s.snaps))

function Base.getindex(s::Snapshots{T},idx) where T
  time_ndofs = num_time_dofs(s)
  nrange = length(idx)
  array = Vector{T}(undef,time_ndofs*nrange)
  for (i,r) in enumerate(idx)
    for nt in 1:time_ndofs
      array[(i-1)*time_ndofs+nt] = s.snaps[nt][r]
    end
  end
  return NonaffinePTArray(array)
end

function Base.vcat(s::Snapshots{T}...) where T
  l = length(first(s))
  vsnaps = Vector{NonaffinePTArray{T}}(undef,l)
  @inbounds for i = 1:l
    vsnaps[i] = vcat(map(n->s[n].snaps[i],eachindex(s))...)
  end
  Snapshots(vsnaps)
end

function save(rbinfo::RBInfo,s::Snapshots)
  path = joinpath(rbinfo.fe_path,"fesnaps")
  save(path,s)
end

function load(rbinfo::RBInfo,T::Type{Snapshots{S}}) where S
  path = joinpath(rbinfo.fe_path,"fesnaps")
  load(path,T)
end
