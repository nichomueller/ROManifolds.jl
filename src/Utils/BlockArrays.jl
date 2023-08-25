function Base.hcat(a::BlockVector,b::BlockVector)
  c = map(hcat,a.blocks,b.blocks)
  d = reshape(c,:,length(c))
  mortar(d)
end

function tpod(bmat::BlockMatrix;kwags...)
  svd_blocks = map(tpod,bmat.blocks)
  mortar(svd_blocks)
end
