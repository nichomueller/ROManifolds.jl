struct DistributedBlockSnapshots{T}
  snaps::Vector{PVector{Vector{PTArray{T}}}}
end

function RB.BlockSnapshots(snaps::Vector{<:PVector{<:Vector{<:PTArray{T}}}}) where T
  DistributedBlockSnapshots(snaps)
end
