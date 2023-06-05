abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct ParamSpace
  domain::Vector{Vector{Float}}
  sampling_style::SamplingStyle
  function ParamSpace(d::Vector{Vector{T}},s::SamplingStyle) where T
    new([Float.(di) for di = d],s)
  end
end

function generate_param(d::Vector{<:Number},::UniformSampling)
  rand(Uniform(first(d),last(d)))
end

function generate_param(d::Vector{<:Number},::UniformSampling)
  rand(Normal(first(d),last(d)))
end

function generate_param(pspace::ParamSpace)
  [generate_param(d,pspace.sampling_style) for d = pspace.domain]
end

struct Param
  p::Vector{Float}
  function Param(p::Vector{Number})
    new(Float.(p))
  end
end

get_param(μ::Param) = μ.p

Base.getindex(μ::Param,args...) = getindex(μ.p,args...)

realization(pspace::ParamSpace) = Param(generate_param(pspace))

realization(pspace::ParamSpace,n::Int) = Table([realization(pspace) for _ = 1:n])

# save(path::String,μ::Table) = save(joinpath(path,"param"),μ)

# function load(::Type{Table{Float,Vector{Param},Vector{Int32}}},path::String)
#   load(joinpath(path,"param"))::Table{Float,Vector{Param},Vector{Int32}}
# end
