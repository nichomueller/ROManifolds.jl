abstract type AbstractSnapshots{T} end

struct Snapshots{T} <: AbstractSnapshots{T}
  snaps::Vector{PTArray{T}}
  Snapshots(s::Vector{<:PTArray{T}}) where T = new{T}(s)
end

Base.length(s::Snapshots) = length(s.snaps)
Base.size(s::Snapshots{T},args...) where {T<:AbstractArray} = size(testitem(first(s.snaps)),args...)
num_space_dofs(s::Snapshots) = size(s,1)
num_time_dofs(s::Snapshots) = length(s)
num_params(s::Snapshots) = length(first(s.snaps))

function Base.getindex(s::Snapshots{T},range) where T
  time_ndofs = length(s)
  nparams = length(range)
  array = Vector{T}(undef,time_ndofs*nparams)
  for nt in 1:time_ndofs
    for np in range
      array[(nt-1)*time_ndofs+np] = s.snaps[nt][np]
    end
  end
  return PTArray(array)
end

function Base.convert(::Type{PTArray{T}},a::Snapshots{T}) where T
  arrays = vcat(map(get_array,a.snaps)...)
  PTArray(arrays)
end

function recenter(
  solver::PThetaMethod,
  s::Snapshots{T},
  μ::Table) where T

  uh0 = solver.uh0
  u0 = get_free_dof_values(uh0(μ))
  sθ = s.snaps*θ + [u0,s.snaps[2:end]...]*(1-θ)
  Snapshots(sθ)
end

struct BlockSnapshots{T} <: AbstractSnapshots{T}
  blocks::Vector{Vector{PTArray{T}}}
  BlockSnapshots(blocks::Vector{Vector{<:PTArray{T}}}) where T = new{T}(blocks)
end

Snapshots(s::Vector{Vector{<:PTArray{T}}}) where T = BlockSnapshots(s)

Base.getindex(s::BlockSnapshots,i...) = s.blocks[i...]
Base.iterate(s::BlockSnapshots,args...) = iterate(s.blocks,args...)
