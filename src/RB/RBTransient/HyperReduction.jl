struct TransientIntegrationDomain <: AbstractIntegrationDomain{AbstractVector{Int32}}
  indices_space::IntegrationDomain
  indices_time::IntegrationDomain
end

function TransientIntegrationDomain(ispace::AbstractVector,itime::AbstractVector)
  indices_space = IntegrationDomain(ispace)
  indices_time = IntegrationDomain(itime)
  TransientIntegrationDomain(indices_space,indices_time)
end

function RBSteady.IntegrationDomain(indices::Tuple{Vararg{AbstractVector}})
  @check length(indices) == 2
  TransientIntegrationDomain(indices...)
end

Base.size(i::TransientIntegrationDomain) = (2,)
function Base.getindex(i::TransientIntegrationDomain,j::Integer)
  j == 1 ? i.indices_space : i.indices_time
end

get_indices_space(i::TransientIntegrationDomain) = RBSteady.get_indices(i[1])
union_indices_space(i::TransientIntegrationDomain...) = RBSteady.union_indices(getindex.(i,1)...)
get_indices_time(i::TransientIntegrationDomain) = RBSteady.get_indices(i[2])
union_indices_time(i::TransientIntegrationDomain...) = RBSteady.union_indices(getindex.(i,2)...)

function Base.getindex(a::AbstractParamArray,i::TransientIntegrationDomain,range::Range2D)
  entry = zeros(eltype2(a),length(i))
  entries = array_of_consecutive_arrays(entry,param_length(a))
  for ip = param_eachindex(entries)
    for (i,is) in enumerate(indices)
      v = consecutive_getindex(a,is,ip)
      consecutive_setindex!(entries,v,i,ip)
    end
  end
  return entries
end

function Base.getindex(a::ParamSparseMatrix,i::TransientIntegrationDomain,range::Range2D)
  ispace,itime = i.indices_space,i.indices_time
  entry = zeros(eltype2(a),length(ispace),length(itime))
  entries = array_of_consecutive_arrays(entry,param_length(a))
  for ip = param_eachindex(entries), (i,it) in enumerate(itime)
    for (i,is) in enumerate(indices)
      v = param_getindex(s,ip)[is]
      consecutive_setindex!(entries,v,i,ip)
    end
  end
  return entries
end

const TransientHyperReduction{A} = HyperReduction{A,TransientIntegrationDomain}

get_indices_space(a::TransientHyperReduction) = get_indices(get_integration_domain_space(a))
union_indices_space(a::TransientHyperReduction...) = union_indices(get_integration_domain_space.(a)...)
get_indices_time(a::TransientHyperReduction) = get_indices(get_integration_domain_time(a))
union_indices_time(a::TransientHyperReduction...) = union_indices(get_integration_domain_time.(a)...)

const TransientMDEIM{A} = MDEIM{A,TransientIntegrationDomain}

get_integration_domain_space(a::TransientMDEIM) = a.domain_space
get_integration_domain_time(a::TransientMDEIM) = a.domain_time

function RBSteady.reduced_jacobian(
  red::Tuple{Vararg{AbstractReduction}},
  op::TransientRBOperator,
  contribs::Tuple{Vararg{Any}})

  a = ()
  for i in eachindex(contribs)
    a = (a...,reduced_jacobian(red[i],op,contribs[i]))
  end
  return a
end

function RBSteady.project!(cache,a::TransientHyperReduction,b::AbstractParamArray)
  cache = coeff,b̂
  interp = get_interpolation(a)
  ldiv!(coeff,interp,vec(b))
  mul!(b̂,a,coeff)
  return b̂
end

const TupOfAffineContribution = Tuple{Vararg{AffineContribution{T}}} where T

function RBSteady.allocate_coefficient(a::TupOfAffineContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  coeffs = ()
  for (a,b) in zip(a,b)
    coeffs = (coeffs...,allocate_coefficient(a,b))
  end
  return coeffs
end

function RBSteady.allocate_hyper_reduction(a::TupOfAffineContribution,b::TupOfArrayContribution)
  RBSteady.allocate_hyper_reduction(first(a),first(b))
end

function RBSteady.project!(cache,a::TupOfAffineContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  cache = coeff,b̂
  for (a,b,c) in zip(a,b,c)
    project!((c,b̂),a,b)
  end
  return b̂
end
