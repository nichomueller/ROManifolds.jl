function RBSteady.inv_project!(
  cache::HRParamArray,
  a::TupOfAffineContribution,
  b::TupOfArrayContribution)

  hypred = cache.hypred
  coeff = cache.coeff
  inv_project!(hypred,coeff,a,b)
end
