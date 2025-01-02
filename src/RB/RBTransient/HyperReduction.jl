"""
    struct TransientIntegrationDomain <: AbstractIntegrationDomain{AbstractVector{Int32}}
      indices_space::IntegrationDomain
      indices_time::IntegrationDomain
    end

Integration domain for a projection operator in a transient context
"""
struct TransientIntegrationDomain <: AbstractIntegrationDomain{AbstractVector{Int32}}
  indices_space::IntegrationDomain
  indices_time::IntegrationDomain
end

function TransientIntegrationDomain(ispace::AbstractVector,itime::AbstractVector)
  indices_space = IntegrationDomain(ispace)
  indices_time = IntegrationDomain(itime)
  TransientIntegrationDomain(indices_space,indices_time)
end

function RBSteady.integration_domain(indices::Union{Tuple,AbstractVector{<:AbstractVector}})
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

function Base.getindex(a::AbstractParamArray,i::TransientIntegrationDomain)
  @notimplemented
end

const TransientHyperReduction{A<:Reduction,B<:ReducedProjection} = HyperReduction{A,B,TransientIntegrationDomain}

get_integration_domain_space(a::TransientHyperReduction) = @abstractmethod
get_integration_domain_time(a::TransientHyperReduction) = @abstractmethod

get_indices_space(a::TransientHyperReduction) = RBSteady.get_indices(get_integration_domain_space(a))
get_indices_time(a::TransientHyperReduction) = RBSteady.get_indices(get_integration_domain_time(a))

union_indices_space(a::TransientHyperReduction...) = union(get_indices_space.(a)...)
union_indices_time(a::TransientHyperReduction...) = union(get_indices_time.(a)...)

union_indices_space(a::AffineContribution) = union_indices_space(get_values(a)...)
union_indices_time(a::AffineContribution) = union_indices_time(get_values(a)...)

function RBSteady.reduced_triangulation(trian::Triangulation,b::TransientHyperReduction,r::RBSpace...)
  indices = get_integration_domain_space(b)
  RBSteady.reduced_triangulation(trian,indices,r...)
end

function RBSteady.HyperReduction(
  red::TransientMDEIMReduction,
  s::Snapshots,
  trial::RBSpace,
  test::RBSpace)

  reduction = get_reduction(red)
  basis = projection(reduction,s)
  proj_basis = project(test,basis,trial,get_combine(red))
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = integration_domain(indices)
  return MDEIM(reduction,proj_basis,factor,domain)
end

const TransientMDEIM{A,B} = MDEIM{A,B,TransientIntegrationDomain}

get_integration_domain_space(a::TransientMDEIM) = get_integration_domain_space(a.domain)
get_integration_domain_time(a::TransientMDEIM) = get_integration_domain_time(a.domain)

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
  cache,
  a::TransientHyperReduction{<:TransientReduction},
  b::AbstractParamArray)

  coeff,b̂ = cache
  interp = RBSteady.get_interpolation(a)
  ldiv!(coeff,interp,vec(b))
  muladd!(b̂,a,coeff)
  return b̂
end

const TupOfAffineContribution = Tuple{Vararg{AffineContribution}}

union_indices_space(a::TupOfAffineContribution) = union(union_indices_space.(a)...)
union_indices_time(a::TupOfAffineContribution) = union(union_indices_time.(a)...)

function RBSteady.allocate_coefficient(a::TupOfAffineContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  coeffs = ()
  for (a,b) in zip(a,b)
    coeffs = (coeffs...,RBSteady.allocate_coefficient(a,b))
  end
  return coeffs
end

function RBSteady.inv_project!(cache,a::TupOfAffineContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  coeff,b̂ = cache
  fill!(b̂,zero(eltype(b̂)))
  for (ai,bi,ci) in zip(a,b,coeff)
    for (aval,bval,cval) in zip(get_values(ai),get_values(bi),get_values(ci))
      inv_project!((cval,b̂),aval,bval)
    end
  end
  return b̂
end

function RBSteady.inv_project!(cache::HRParamArray,a::TupOfAffineContribution,b::TupOfArrayContribution)
  coeff = cache.coeff
  hypred = cache.hypred
  inv_project!((coeff,hypred),a,b)
end

function RBSteady.allocate_hypred_cache(a::TupOfAffineContribution,r::TransientRealization)
  coeffs = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  hypred = RBSteady.allocate_hyper_reduction(first(a),r)
  return coeffs,hypred
end

# multi field interface

for f in (:get_indices_space,:get_indices_time)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::HyperReduction)
      cache = $f(a)
      return cache
    end

    function Arrays.return_cache(::typeof($f),a::BlockHyperReduction)
      i = findfirst(a.touched)
      @notimplementedif isnothing(i)
      cache = return_cache($f,a[i])
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHyperReduction)
      cache = return_cache($f,a)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i])
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
end

function union_indices_space(a::BlockHyperReduction...)
  @check all(ai.touched == a[1].touched for ai in a)
  cache = return_cache(get_indices_space,first(a))
  for ai in a
    for i in eachindex(ai)
      if ai.touched[i]
        if isassigned(cache,i)
          cache[i] = union(cache[i],get_indices_space(ai[i]))
        else
          cache[i] = get_indices_space(ai[i])
        end
      end
    end
  end
  ArrayBlock(cache,a[1].touched)
end

function union_indices_time(a::BlockHyperReduction...)
  cache = Vector{Int32}[]
  for ai in a
    for i in eachindex(ai)
      if ai.touched[i]
        push!(cache,get_indices_time(ai[i]))
      end
    end
  end
  union(cache...)
end
