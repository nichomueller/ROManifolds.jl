function RBSteady.reduction(red::PODReduction,A::TransientSparseSnapshots,args...)
  red_style = ReductionStyle(red)
  U,S,V = tpod(red_style,A,args...)
  return recast(U,A)
end

function RBSteady.reduction(red::TTSVDReduction,A::TransientSparseSnapshots,args...)
  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,args...)
  return recast(cores,A)
end
