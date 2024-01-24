struct DistributedBlockSnapshots{T}
  snaps::Vector{PVector{Vector{ParamArray{T}}}}
end

function RB.BlockSnapshots(snaps::Vector{<:PVector{<:Vector{<:ParamArray{T}}}}) where T
  DistributedBlockSnapshots(snaps)
end
