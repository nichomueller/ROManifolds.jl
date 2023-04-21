struct ProblemType
  steady::Bool
  indef::Bool
  pdomain::Bool
end

issteady(p::ProblemType) = Val(p.steady)

isindef(p::ProblemType) = Val(p.indef)

ispdomain(p::ProblemType) = Val(p.pdomain)

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct ParamSpace
  domain::Vector{<:Vector{<:Number}}
  sampling_style::SamplingStyle
end

generate_param(d::Vector{<:Number},::UniformSampling) = rand(Uniform(first(d),last(d)))

generate_param(d::Vector{<:Number},::NormalSampling) = rand(Normal(first(get_μ(d)),last(get_μ(d))))

generate_param(P::ParamSpace) = [generate_param(d,P.sampling_style) for d = P.domain]

struct Param
  μ::Vector{Number}
end

get_μ(p::Param) = p.μ

realization(P::ParamSpace) = Param(generate_param(P))

Base.getindex(p::Param,args...) = getindex(p.μ,args...)

Base.Matrix(pvec::Vector{Param}) = Matrix{Float}(reduce(vcat,transpose.(getproperty.(pvec,:μ)))')

Distributions.var(p::Param) = var(get_μ(p))

Base.:(-)(p1::Param,p2::Param) = get_μ(p1) .- get_μ(p2)

struct ParamVecs
  pvec::Vector{Param}
end

realization(P::ParamSpace,n) = ParamVecs([realization(P) for _ = 1:n])

save(path::String,pvec::Vector{Param}) = save(joinpath(path,"param"),Matrix(pvec))

function load_param(::Type{ParamVecs},path::String)
  param_mat = load(joinpath(path,"param"))
  param_block = vblocks(param_mat)
  ParamVecs(Param.(param_block))
end

abstract type ProblemMeasures end

struct ProblemFixedMeasures <: ProblemMeasures
  dΩ::Measure
  dΓn::Measure
end

struct ProblemParamMeasures <: ProblemMeasures
  dΩ::Function
  dΓn::Function
end

function ProblemMeasures(model::DiscreteModel,order=1)
  degree = get_degree(order)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ProblemFixedMeasures(dΩ,dΓn)
end

function ProblemMeasures(model::Function,order=1)
  degree = get_degree(order)
  Ω(μ) = Triangulation(model(μ))
  dΩ(μ) = Measure(Ω(μ),degree)
  Γn(μ) = BoundaryTriangulation(model(μ),tags=["neumann"])
  dΓn(μ) = Measure(Γn(μ),degree)

  ProblemFixedMeasures(dΩ,dΓn)
end

get_dΩ(meas::ProblemFixedMeasures) = meas.dΩ

get_dΓn(meas::ProblemFixedMeasures) = meas.dΓn

get_dΩ(meas::ProblemParamMeasures,p::Param) = meas.dΩ(p)

get_dΓn(meas::ProblemParamMeasures,p::Param) = meas.dΓn(p)
