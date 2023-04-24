mutable struct Snapshots
  id::Symbol
  snap::EMatrix{Float}
  nsnap::Int
end

function Snapshots(
  id::Symbol,
  snap::EMatrix{Float},
  nsnap=get_nsnap(snap))

  Snapshots(id,snap,nsnap)
end

function Snapshots(id::Symbol,snap::Matrix{Float},args...)
  Snapshots(id,EMatrix(snap),args...)
end

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

get_snap(s::Snapshots) = s.snap

get_nsnap(s::Snapshots) = s.nsnap

get_nsnap(v::AbstractVector) = length(v)

get_nsnap(m::AbstractMatrix) = size(m,2)

save(path::String,s::Snapshots) = save(joinpath(path,"$(s.id)"),s.snap)

function load(path::String,id::Symbol,nsnap::Int)
  s = load(EMatrix{Float},joinpath(path,"$(id)"))
  Snapshots(id,s,nsnap)
end

get_Nt(s::Snapshots) = get_Nt(get_snap(s),get_nsnap(s))

mode2_unfolding(s::Snapshots) = mode2_unfolding(get_snap(s),get_nsnap(s))

POD(s::Snapshots,args...;kwargs...) = POD(s.snap,args...;kwargs...)

function Gridap.FESpaces.FEFunction(fespace,u::Snapshots,args...)
  FEFunction(fespace,get_snap(u),args...)
end

function compute_in_timesθ(snaps::Snapshots,args...)
  id = get_id(snaps)*:θ
  snap = get_snap(snaps)
  nsnap = get_nsnap(snaps)
  Snapshots(id,compute_in_timesθ(snap,args...),nsnap)
end

function Base.vcat(s1::Snapshots,s2::Snapshots)
  @assert get_nsnap(s1) == get_nsnap(s2) "Cannot concatenate input snapshots"
  id = get_id(s1)*get_id(s2)
  snap = vcat(get_snap(s1),get_snap(s2))
  nsnap = get_nsnap(s1)
  Snapshots(id,snap,nsnap)
end
