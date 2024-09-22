struct TransientIntegrationDomain <: AbstractIntegrationDomain{AbstractVector{Int32}}
  indices_space::IntegrationDomain
  indices_time::IntegrationDomain
end

function TransientIntegrationDomain(ispace::AbstractVector,itime::AbstractVector)
  indices_space = IntegrationDomain(ispace)
  indices_time = IntegrationDomain(itime)
  TransientIntegrationDomain(indices_space,indices_time)
end

function RBSteady.integration_domain(indices::Tuple{Vararg{AbstractVector}})
  @check length(indices) == 2
  TransientIntegrationDomain(indices...)
end

Base.size(i::TransientIntegrationDomain) = (2,)
function Base.getindex(i::TransientIntegrationDomain,j::Integer)
  j == 1 ? i.indices_space : i.indices_time
end

get_integration_domain_space(i::TransientIntegrationDomain) = i.indices_space
get_integration_domain_time(i::TransientIntegrationDomain) = i.indices_time
get_indices_space(i::TransientIntegrationDomain) = RBSteady.get_indices(i[1])
union_indices_space(i::TransientIntegrationDomain...) = RBSteady.union_indices(getindex.(i,1)...)
get_indices_time(i::TransientIntegrationDomain) = RBSteady.get_indices(i[2])
union_indices_time(i::TransientIntegrationDomain...) = RBSteady.union_indices(getindex.(i,2)...)

const TransientHyperReduction{A} = HyperReduction{A,TransientIntegrationDomain}

get_integration_domain_space(a::TransientHyperReduction) = @abstractmethod
get_integration_domain_time(a::TransientHyperReduction) = @abstractmethod

get_indices_space(a::TransientHyperReduction) = RBSteady.get_indices(get_integration_domain_space(a))
union_indices_space(a::TransientHyperReduction...) = RBSteady.union_indices(get_integration_domain_space.(a)...)
get_indices_time(a::TransientHyperReduction) = RBSteady.get_indices(get_integration_domain_time(a))
union_indices_time(a::TransientHyperReduction...) = RBSteady.union_indices(get_integration_domain_time.(a)...)

union_indices_space(a::AffineContribution) = union_indices_space(get_values(a)...)
union_indices_time(a::AffineContribution) = union_indices_time(get_values(a)...)

function RBSteady.reduced_triangulation(trian::Triangulation,b::TransientHyperReduction,r::FESubspace...)
  indices = get_integration_domain_space(b)
  RBSteady.reduced_triangulation(trian,indices,r...)
end

function RBSteady.HyperReduction(
  red::TransientMDEIMReduction,
  s::AbstractSnapshots,
  trial::FESubspace,
  test::FESubspace)

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_combine(red))
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = integration_domain(indices)
  return MDEIM(proj_basis,factor,domain)
end

const TransientMDEIM{A} = MDEIM{A,TransientIntegrationDomain}

get_integration_domain_space(a::TransientMDEIM) = get_integration_domain_space(a.integration_domain)
get_integration_domain_time(a::TransientMDEIM) = get_integration_domain_time(a.integration_domain)

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

function RBSteady.allocate_coefficient(a::TransientHyperReduction,r::TransientRealization)
  RBSteady.allocate_coefficient(a,param_length(r))
end

function RBSteady.allocate_hyper_reduction(a::TransientHyperReduction,r::TransientRealization)
  RBSteady.allocate_hyper_reduction(a,param_length(r))
end

const TupOfAffineContribution = Tuple{Vararg{AffineContribution}}

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
