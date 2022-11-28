abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct ProblemType
  steady::Bool
  indef::Bool
  pdomain::Bool
end

issteady(p::ProblemType) = Val(p.steady)
isindef(p::ProblemType) = Val(p.indef)
ispdomain(p::ProblemType) = Val(p.pdomain)

struct Param
  param::Vector{Float}
end

struct ParamSpace
  domain::Vector{Param}
  sampling_style::SamplingStyle
end

generate_param(d::Param,::UniformSampling) = rand(Uniform(first(d.param),last(d.param)))
generate_param(d::Param,::NormalSampling) = rand(Normal(first(d.param),last(d.param)))
generate_param(P::ParamSpace) = Broadcasting(d->generate_param(d,P.sampling_style))(P.domain)
generate_param(P::ParamSpace,n::Int) = [generate_param(P) for _ = 1:n]
realization(P::ParamSpace) = Param(generate_param(P))
realization(P::ParamSpace,n) = Param.(generate_param(P,n))
get_μ(p::Param) = p.param

mutable struct ParamFunction{S}
  id::Symbol
  pspace::ParamSpace
  f::Function

  function ParamFunction(
    ::ProblemType{I,S,M},
    id::Symbol,
    pspace::ParamSpace,
    f::Function) where {I,S,M}
    new{S}(id,pspace,f)
  end
end

function realization(fμ::ParamFunction)
  μ = realization(fμ.pspace)
  μ,fμ.f(μ)
end

Base.zero(::Type{Param}) = 0.
Base.iterate(p::Param,i = 1) = iterate(p.param,i)
Base.getindex(p::Param,args...) = getindex(p.param,args...)
