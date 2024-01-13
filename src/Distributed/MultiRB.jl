struct DistributedBlockSnapshots{T}
  snaps::Vector{PVector{Vector{PArray{T}}}}
end

function RB.BlockSnapshots(snaps::Vector{<:PVector{<:Vector{<:PArray{T}}}}) where T
  DistributedBlockSnapshots(snaps)
end
