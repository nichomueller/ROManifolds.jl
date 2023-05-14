struct ProblemType
  steady::Bool
  indef::Bool
  pdomain::Bool
end

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

realization(P::ParamSpace,n) = Vector{Param}([realization(P) for _ = 1:n])

Base.getindex(p::Param,args...) = getindex(p.μ,args...)

LinearAlgebra.Matrix(pvec::Vector{Param}) = Matrix{Float}(reduce(vcat,transpose.(getproperty.(pvec,:μ)))')

Distributions.var(p::Param) = var(get_μ(p))

Base.:(-)(p1::Param,p2::Param) = get_μ(p1) .- get_μ(p2)

function collect_param_from_workers()
  param_mat = collect_from_workers(Matrix{Float},:μ)
  param_block = vblocks(param_mat)
  Vector{Param}(Param.(param_block))
end

save(path::String,pvec::Vector{Param}) = save(joinpath(path,"param"),Matrix(pvec))

function load(::Type{Vector{Param}},path::String)
  param_mat = load(joinpath(path,"param"))
  param_block = vblocks(param_mat)
  Vector{Param}(Param.(param_block))
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
