struct Snapshots{T<:AbstractArray}
  snaps::Vector{PTArray{T}}
  function Snapshots(s::Vector{<:PTArray{T}}) where T
    new{T}(s)
  end
end

Base.length(s::Snapshots) = length(s.snaps)
Base.size(s::Snapshots,args...) = size(testitem(first(s.snaps)),args...)
Base.eachindex(s::Snapshots) = eachindex(s.snaps)
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

function Base.convert(::Type{PTArray{T}},a::Snapshots{T}) where T
  arrays = vcat(map(get_array,a.snaps)...)
  PTArray(arrays)
end

function recenter(
  fesolver::PThetaMethod,
  s::Snapshots{T},
  μ::Table) where T

  θ = fesolver.θ
  uh0 = fesolver.uh0(μ)
  u0 = get_free_dof_values(uh0)
  sθ = s.snaps.*θ + [u0,s.snaps[2:end]...].*(1-θ)
  Snapshots(sθ)
end

function save(info::RBInfo,s::Snapshots)
  if info.save_solutions
    path = joinpath(info.fe_path,"fesnaps")
    save(path,s)
  end
end

function load(info::RBInfo,T::Type{<:Snapshots})
  path = joinpath(info.fe_path,"fesnaps")
  load(path,T)
end
