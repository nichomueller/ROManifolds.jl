function RBSteady.HyperReduction(
  red::TransientMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace)

  reduction = get_reduction(red)
  basis = projection(reduction,s)
  proj_basis = project(test,basis,trial,get_combine(red))
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(trian,trial,test,indices...)
  return MDEIM(reduction,proj_basis,factor,domain)
end

const TransientHyperReduction{A<:Reduction,B<:ReducedProjection} = HyperReduction{A,B,TransientIntegrationDomain}

function get_integration_domain_space(a::TransientHyperReduction)
  i = get_integration_domain(a)
  i.domain_space
end

function get_indices_time(a::TransientHyperReduction)
  i = get_integration_domain(a)
  i.indices_time
end

function RBSteady.reduced_triangulation(trian::Triangulation,a::TransientHyperReduction)
  reduced_triangulation(trian,get_integration_domain_space(a))
end

function RBSteady.reduced_jacobian(
  red::Tuple{Vararg{Reduction}},
  trial::RBSpace,
  test::RBSpace,
  contribs::Tuple{Vararg{Any}})

  a = ()
  for i in eachindex(contribs)
    a = (a...,reduced_jacobian(red[i],trial,test,contribs[i]))
  end
  return a
end

function RBSteady.inv_project!(
  b̂::AbstractParamArray,
  coeff::AbstractParamArray,
  a::TransientHyperReduction{<:TransientReduction},
  b::AbstractParamArray)

  o = one(eltype2(b̂))
  interp = RBSteady.get_interpolation(a)
  ldiv!(coeff,interp,vec(b))
  mul!(b̂,a,coeff,o,o)
  return b̂
end

const TupOfAffineContribution = Tuple{Vararg{AffineContribution}}

function RBSteady.allocate_coefficient(a::TupOfAffineContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  coeffs = ()
  for (a,b) in zip(a,b)
    coeffs = (coeffs...,RBSteady.allocate_coefficient(a,b))
  end
  return coeffs
end

function RBSteady.inv_project!(
  b̂::AbstractParamArray,
  coeff::TupOfArrayContribution,
  a::TupOfAffineContribution,
  b::TupOfArrayContribution)

  @check length(coeff) == length(a) == length(b)
  fill!(b̂,zero(eltype(b̂)))
  for (ai,bi,ci) in zip(a,b,coeff)
    for (aval,bval,cval) in zip(get_contributions(ai),get_contributions(bi),get_contributions(ci))
      inv_project!(b̂,cval,aval,bval)
    end
  end
  return b̂
end

function RBSteady.allocate_hypred_cache(a::TupOfAffineContribution,r::TransientRealization)
  fecache = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  coeffs = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  hypred = RBSteady.allocate_hyper_reduction(first(a),r)
  return HRParamArray(fecache,coeffs,hypred)
end
