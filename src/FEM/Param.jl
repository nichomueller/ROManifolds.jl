struct ProblemType
  steady::Bool
  indef::Bool
end

isindef(p::ProblemType) = Val{p.indef}()

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

function vector_of_params(param_block::Vector{Vector{Float}})
  Vector{Param}(Param.(param_block))
end

function vector_of_params(param_mat::AbstractMatrix)
  vector_of_params(vblocks(param_mat))
end

realization(P::ParamSpace) = Param(generate_param(P))

realization(P::ParamSpace,n) = [realization(P) for _ = 1:n]

Base.getindex(p::Param,args...) = getindex(p.μ,args...)

LinearAlgebra.Matrix(pvec::Vector{Param}) = Matrix{Float}(reduce(vcat,transpose.(getproperty.(pvec,:μ)))')

save(path::String,pvec::Vector{Param}) = save(joinpath(path,"param"),Matrix(pvec))

function load(::Type{Vector{Param}},path::String)
  vector_of_params(load(joinpath(path,"param")))
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

get_dΩ(meas::ProblemFixedMeasures) = meas.dΩ

get_dΓn(meas::ProblemFixedMeasures) = meas.dΓn
