abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

#= abstract type ParamSpace end

struct MyParamSpace <: ParamSpace
  ρ::Float
  ν::Function
  f::Function

  u::FEFunction
end

struct ParamSample{P<:MyParamSpace}
  cache
end

function realization(P::MyParamSpace)
  uvec = rand(get_free_dof_values(P.u))
  cache = uvec
  ParamSample{P}(cache)
end =#

mutable struct ParamSpace
  domain::Vector{Vector{Float}}
  sampling_style::SamplingStyle
end

realization(d::Vector{Float},::UniformSampling) = rand(Uniform(first(d),last(d)))
realization(d::Vector{Float},::NormalSampling) = rand(Normal(first(d),last(d)))
realization(P::ParamSpace) = Broadcasting(d->realization(d,P.sampling_style))(P.domain)
realization(P::ParamSpace,n::Int) = [realization(P) for _ = 1:n]

mutable struct ParamFunctional{S}
  param_space::ParamSpace
  f::Function

  function ParamFunctional(
    P::ParamSpace,
    f::Function;
    S=false)
    new{S}(P,f)
  end
end

function realization(Fμ::ParamFunctional{S}) where S
  μ = realization(Fμ.param_space)
  μ, Fμ.f(μ)
end

function realization(
  P::ParamSpace,
  f::Function;
  S=false)

  Fμ = ParamFunctional(P,f;S)
  realization(Fμ)
end
