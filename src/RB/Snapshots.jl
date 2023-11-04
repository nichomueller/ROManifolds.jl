struct Snapshots{T<:AbstractArray}
  snaps::Vector{PTArray{T}}
  function Snapshots(s::Vector{<:PTArray{T}}) where T
    new{T}(s)
  end
end

Base.length(s::Snapshots) = length(s.snaps)
Base.size(s::Snapshots,args...) = size(testitem(first(s.snaps)),args...)
Base.eachindex(s::Snapshots) = eachindex(s.snaps)
Base.lastindex(s::Snapshots) = num_params(s)
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
  return PTArray(array)
end

function Base.copy(s::Snapshots)
  scopy = copy.(s.snaps)
  Snapshots(scopy)
end

function Base.vcat(s::Snapshots{T}...) where T
  l = length(first(s))
  vsnaps = Vector{PTArray{T}}(undef,l)
  @inbounds for i = 1:l
    vsnaps[i] = vcat(map(n->s[n].snaps[i],eachindex(s))...)
  end
  Snapshots(vsnaps)
end

function recenter(s::Snapshots,uh0::PTFEFunction;θ::Real=1)
  snaps = copy(s.snaps)
  u0 = get_free_dof_values(uh0)
  sθ = snaps*θ + [u0,snaps[2:end]...]*(1-θ)
  Snapshots(sθ)
end

function nearest_neighbor(sols::Snapshots,params::Table,params_test::Table)
  _sols = copy(sols)
  kdtree = KDTree(map(x -> SVector(Tuple(x)),params))
  idx_dist = map(x -> nn(kdtree,SVector(Tuple(x))),params_test)
  _sols[first.(idx_dist)]
end

function save(info::RBInfo,s::Snapshots)
  path = joinpath(info.fe_path,"fesnaps")
  save(path,s)
end

function load(info::RBInfo,T::Type{<:Snapshots})
  path = joinpath(info.fe_path,"fesnaps")
  load(path,T)
end
