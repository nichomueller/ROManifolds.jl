function RBSteady.HyperReduction(
  red::TransientMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace)

  reduction = get_reduction(red)
  basis = projection(reduction,s)
  proj_basis = project(test,basis)
  (rows,indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(trian,test,rows,indices_time)
  return MDEIM(reduction,proj_basis,factor,domain)
end

function RBSteady.HyperReduction(
  red::TransientMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace)

  reduction = get_reduction(red)
  basis = projection(reduction,s)
  proj_basis = project(test,basis,trial,get_combine(red))
  ((rows,cols),indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(trian,trial,test,rows,cols,indices_time)
  return MDEIM(reduction,proj_basis,factor,domain)
end

const TransientHyperReduction{A<:Reduction,B<:ReducedProjection} = HyperReduction{A,B,TransientIntegrationDomain}

for f in (:get_integration_domain_space,:get_indices_time,:get_itimes)
  @eval begin
    $f(a::TransientHyperReduction) = $f(get_integration_domain(a))
  end
end

get_itimes(a::TransientHyperReduction,args...) = get_itimes(get_integration_domain(a),args...)

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

function RBSteady.inv_project!(cache::HRParamArray,a::TupOfAffineContribution)
  inv_project!(cache.hypred,cache.coeff,a,cache.fecache)
end

function RBSteady.allocate_hypred_cache(a::TupOfAffineContribution,r::TransientRealization)
  fecache = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  coeffs = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  hypred = RBSteady.allocate_hyper_reduction(first(a),r)
  return HRParamArray(fecache,coeffs,hypred)
end

function get_common_time_domain(a::TransientHyperReduction...)
  time_ids = ()
  for ai in a
    time_ids = (time_ids...,get_indices_time(ai))
  end
  union(time_ids...)
end

function get_common_time_domain(a::AffineContribution)
  get_common_time_domain(get_contributions(a)...)
end

function get_common_time_domain(a::TupOfAffineContribution)
  union(map(get_common_time_domain,a)...)
end

function get_param_itimes(a::HyperReduction,common_ids::Range2D)
  common_param_ids = common_ids.axis1
  common_time_ids = common_ids.axis2
  local_time_ids = get_indices_time(a)
  local_itime_ids = get_itimes(a,common_time_ids)
  locations = range_1d(common_param_ids,local_itime_ids,length(common_param_ids))
  return locations
end

function Arrays.return_cache(::typeof(get_indices_time),a::BlockHyperReduction)
  cache = get_indices_time(testitem(a))
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function get_indices_time(a::BlockHyperReduction)
  cache = return_cache(get_itimes,a)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_itimes(a[i])
    end
  end
  return ArrayBlock(cache,a.touched)
end

for f in (:get_itimes,:get_param_itimes)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::BlockHyperReduction,ids::AbstractVector)
      cache = $f(testitem(a),ids)
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHyperReduction,ids::AbstractVector)
      cache = return_cache($f,a,ids)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i],ids)
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
end

function get_common_time_domain(a::BlockHyperReduction...)
  time_ids = ()
  for ai in a
    for i in eachindex(ai)
      if ai.touched[i]
        time_ids = (time_ids...,get_indices_time(ai[i]))
      end
    end
  end
  union(time_ids...)
end

for f in (:get_itimes,:get_param_itimes), T in (:HyperReduction,:BlockHyperReduction)
  @eval begin
    function $f(a::$T,common_ids::Range1D)
      $f(a,common_ids.parent)
    end
  end
end
