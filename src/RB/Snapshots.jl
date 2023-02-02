mutable struct Snapshots{T}
  id::Symbol
  snap::AbstractArray{T}
  nsnap::Int
end

function Snapshots(id::Symbol,snap::AbstractArray{T},nsnap::Int) where {T<:DataType}
  Snapshots{T}(id,snap,nsnap)
end

function Snapshots(id::Symbol,snap::AbstractArray{T}) where {T<:DataType}
  nsnap = get_nsnap(snap)
  Snapshots{T}(id,snap,nsnap)
end

function Snapshots(id::Symbol,blocks::Vector{<:AbstractArray{T}}) where T
  snap = Matrix(blocks)
  nsnap = get_nsnap(blocks)
  Snapshots{T}(id,snap,nsnap)
end

function Snapshots(id::Symbol,snap::NTuple{N,AbstractArray}) where N
  Broadcasting(si->Snapshots(id,si))(snap)
end

function Base.getindex(s::Snapshots,idx::Int)
  Nt = get_Nt(s)
  snap_idx = s.snap[:,(idx-1)*Nt+1:idx*Nt]
  Snapshots(s.id,snap_idx,1)
end

function Base.getindex(s::Snapshots,idx::UnitRange{Int})
  Nt = get_Nt(s)
  snap(i) = getindex(s.snap,:,(i-1)*Nt+1:i*Nt)
  Snapshots(s.id,Matrix(snap.(idx)),length(idx))
end

get_id(s::Snapshots) = s.id

get_snap(s::Snapshots) = s.snap

get_snap(s::NTuple{N,Snapshots}) where N = get_snap.(s)

get_nsnap(s::Snapshots) = s.nsnap

get_nsnap(v::AbstractVector) = length(v)

get_nsnap(m::AbstractMatrix) = size(m)[2]

save(path::String,s::Snapshots) = save(joinpath(path,"$(s.id)"),s.snap)

function load_snap(path::String,id::Symbol,nsnap::Int)
  s = load(joinpath(path,"$(id)"))
  Snapshots(id,s,nsnap)
end

get_Nt(s::Snapshots) = get_Nt(get_snap(s),get_nsnap(s))

mode2_unfolding(s::Snapshots) = mode2_unfolding(get_snap(s),get_nsnap(s))

mode2_unfolding(s::NTuple{N,Snapshots}) where N = mode2_unfolding.(s)

POD(s::Snapshots,args...;kwargs...) = POD(s.snap,args...;kwargs...)

POD(s::NTuple{N,Snapshots},args...;kwargs...) where N = Broadcasting(si->POD(si,args...;kwargs...))(s)

function SparseArrays.findnz(s::Snapshots)
  id,snap,nsnap = get_id(s),get_snap(s),get_nsnap(s)
  i,j,v = findnz(sparse(snap))
  snap_nnz = reshape(v,:,get_Nt(s)*nsnap)
  i,j,Snapshots(id,snap_nnz,nsnap)
end

function Gridap.FESpaces.FEFunction(fespace,u::Snapshots,args...)
  FEFunction(fespace,get_snap(u),args...)
end

function compute_in_timesθ(snaps::Snapshots,args...;kwargs...)
  id = get_id(snaps)*:θ
  snap = get_snap(snaps)
  nsnap = get_nsnap(snaps)
  Snapshots(id,compute_in_timesθ(snap,args...;kwargs...),nsnap)
end
