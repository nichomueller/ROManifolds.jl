abstract type AbstractSnapshots{T} end

struct Snapshots{T} <: AbstractSnapshots{T}
  s::Vector{PTArray{T}}
  Snapshots(s::Vector{<:PTArray{T}}) where T = new{T}(s)
end

function recenter(
  solver::PThetaMethod,
  s::Snapshots{T},
  μ::Table) where T

  uh0 = solver.uh0
  u0 = get_free_dof_values(uh0(μ))
  sθ = s.snaps*θ + [u0,s.snaps[2:end]...]*(1-θ)
  to_ptarray(PTArray{T},sθ)
end

function Base.getindex(s::Snapshots{T},range) where T
  time_ndofs = length(s)
  nparams = length(range)
  array = Vector{T}(undef,time_ndofs*nparams)
  for nt in eachindex(s)
    for np in range
      array[(nt-1)*time_ndofs+np] = s[nt][np]
    end
  end
  return PTArray(array)
end

struct BlockSnapshots{T} <: AbstractSnapshots{T}
  blocks::Vector{Vector{PTArray{T}}}
  BlockSnapshots(blocks::Vector{Vector{<:PTArray{T}}}) where T = new{T}(blocks)
end

Snapshots(s::Vector{Vector{<:PTArray{T}}}) where T = BlockSnapshots(s)

Base.getindex(s::BlockSnapshots,i...) = s.blocks[i...]
Base.iterate(s::BlockSnapshots,args...) = iterate(s.blocks,args...)
