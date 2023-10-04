abstract type AbstractSnapshots{T} end

struct Snapshots{T<:AbstractArray} <: AbstractSnapshots{T}
  snaps::Vector{PTArray{T}}
  function Snapshots(s::Vector{<:PTArray{T}}) where T
    r = reorder(s)
    new{T}(r)
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
  nidx = length(idx)
  array = Vector{T}(undef,time_ndofs*nidx)
  for (count,i) in enumerate(idx)
    for nt in 1:time_ndofs
      @inbounds array[(count-1)*time_ndofs+nt] = s.snaps[i][nt]
    end
  end
  return PTArray(array)
end

function Base.convert(::Type{PTArray{T}},a::Snapshots{T}) where T
  arrays = vcat(map(get_array,a.snaps)...)
  PTArray(arrays)
end

function reorder(s::Vector{PTArray{T}}) where T
  time_ndofs = length(s)
  s1 = testitem(s)
  nparams = length(s1)

  array = Vector{T}(undef,time_ndofs)
  r = Vector{PTArray{T}}(undef,nparams)
  @inbounds for np in 1:nparams
    for nt in 1:time_ndofs
      array[nt] = s[nt][np]
    end
    r[np] = PTArray(copy(array))
  end

  return r
end

function recenter(
  solver::PThetaMethod,
  s::Snapshots{T},
  μ::Table) where T

  θ = solver.θ
  uh0 = solver.uh0
  u0 = get_free_dof_values(uh0(μ))
  sθ = s.snaps.*θ + [u0,s.snaps[2:end]...].*(1-θ)
  Snapshots(sθ)
end

struct BlockSnapshots{T} <: AbstractSnapshots{T}
  blocks::Vector{Vector{PTArray{T}}}
  BlockSnapshots(blocks::Vector{Vector{<:PTArray{T}}}) where T = new{T}(blocks)
end

Snapshots(s::Vector{Vector{<:PTArray{T}}}) where T = BlockSnapshots(s)

Base.getindex(s::BlockSnapshots,i...) = s.blocks[i...]
Base.iterate(s::BlockSnapshots,args...) = iterate(s.blocks,args...)

function save(info::RBInfo,nzm::AbstractSnapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps")
    save(path,nzm)
  end
end

function load(info::RBInfo,T::Type{<:AbstractSnapshots})
  path = joinpath(info.fe_path,"fesnaps")
  load(path,T)
end
