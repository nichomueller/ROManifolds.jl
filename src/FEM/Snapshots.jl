mutable struct Snapshots
  id::Symbol
  snap::AbstractMatrix{Float}
  nsnap::Int
end

function Snapshots(
  id::Symbol,
  snap::AbstractMatrix{Float},
  nsnap=size(snap,2),
  convert_type=nothing)

  if isnothing(convert_type)
    Snapshots(id,snap,nsnap)
  else typeof(convert_type) == DataType
    Snapshots(id,convert(convert_type,snap),nsnap)
  end
end

#= function convert_snapshot(::Type{T},s::Snapshots) where T
  id = get_id(s)
  snap = get_snap(s)
  nsnap = get_nsnap(s)
  Snapshots(id,convert(T,snap),nsnap)
end =#

function Base.getindex(s::Snapshots,idx::UnitRange{Int})
  nidx = length(idx)
  nsnap = get_nsnap(s)
  Nt = get_Nt(s)
  @assert nidx ≤ get_nsnap(s) "BoundsError: attempt to access Snapshot with nsnap = $nsnap at idx = $nidx"
  idx1,idxN = first(idx),last(idx)
  Snapshots(s.id,get_snap(s)[:,(idx1-1)*Nt+1:idxN*Nt],length(idx))
end

Base.getindex(s::Snapshots,idx::Int) = getindex(s,idx:idx)

get_id(s::Snapshots) = s.id

get_id(s::NTuple{N,Snapshots}) where N = get_id.(s)

get_snap(s::Snapshots) = s.snap

get_snap(::Type{T},s::Snapshots) where T = convert(T,s.snap)

get_nsnap(s::Snapshots) = s.nsnap

get_nsnap(mat::AbstractMatrix,Nt::Int) = Int(size(mat,2)/Nt)

function save(path::String,s::Snapshots)
  smat = get_snap(Matrix{Float},s)
  save(joinpath(path,"$(get_id(s))"),smat)
end

function load(path::String,id::Symbol,nsnap::Int)
  s = load(Matrix{Float},joinpath(path,"$(id)"))
  Snapshots(id,s,nsnap)
end

get_Nt(s::Snapshots) = get_Nt(get_snap(s),get_nsnap(s))

mode2_unfolding(s::Snapshots) = mode2_unfolding(get_snap(s),get_nsnap(s))

POD(s::Snapshots,args...;kwargs...) = POD(s.snap,args...;kwargs...)

function Gridap.FESpaces.FEFunction(fespace,u::Snapshots,args...)
  FEFunction(fespace,get_snap(u),args...)
end

function compute_in_times(snaps::Snapshots,args...)
  id = get_id(snaps)*:θ
  snap = get_snap(snaps)
  nsnap = get_nsnap(snaps)
  Snapshots(id,compute_in_times(snap,args...),nsnap)
end

function Base.vcat(s1::Snapshots,s2::Snapshots)
  @assert get_nsnap(s1) == get_nsnap(s2) "Cannot concatenate input snapshots"
  id = get_id(s1)*get_id(s2)
  snap = vcat(get_snap(s1),get_snap(s2))
  nsnap = get_nsnap(s1)
  Snapshots(id,snap,nsnap)
end
