@inline slow_index(i,N::Integer) = cld.(i,N)
@inline slow_index(i::Colon,::Integer) = i
@inline fast_index(i,N::Integer) = mod.(i .- 1,N) .+ 1
@inline fast_index(i::Colon,::Integer) = i

abstract type AbstractParamRealization end

struct ParamRealization{P<:AbstractVector} <: AbstractParamRealization
  params::P
end

const TrivialParamRealization = ParamRealization{<:AbstractVector{<:Real}}

get_params(r::ParamRealization) = r # we only want to deal with a ParamRealization type
_get_params(r::ParamRealization) = r.params # this function should stay local
_get_params(r::TrivialParamRealization) = [r.params] # this function should stay local
num_params(r::ParamRealization) = length(_get_params(r))
Base.length(r::ParamRealization) = num_params(r)
Base.getindex(r::ParamRealization,i) = ParamRealization(getindex(_get_params(r),i))
Base.copy(r::ParamRealization) = ParamRealization(copy(_get_params(r)))
Arrays.testitem(r::ParamRealization) = testitem(_get_params(r))

# when iterating over a ParamRealization{P}, we return eltype(P) ∀ index i
function Base.iterate(r::ParamRealization,state=1)
  if state > length(r)
    return nothing
  end
  rstate = _get_params(r)[state]
  return rstate, state+1
end

# Convention: we separate the initial time instant t0 from the others:
# in unsteady FEM applications, the value of the solution at t0 is given
abstract type TransientParamRealization{P<:ParamRealization,T<:Real} <: AbstractParamRealization end

Base.length(r::TransientParamRealization) = num_params(r)*num_times(r)
get_params(r::TransientParamRealization) = get_params(r.params)
_get_params(r::TransientParamRealization) = _get_params(r.params)
num_params(r::TransientParamRealization) = num_params(r.params)
num_times(r::TransientParamRealization) = length(get_times(r))

struct GenericTransientParamRealization{P,T,A} <: TransientParamRealization{P,T}
  params::P
  times::A
  t0::T
end

function TransientParamRealization(params::ParamRealization,times::AbstractVector{<:Real},t0::Real)
  GenericTransientParamRealization(params,times,t0)
end

function TransientParamRealization(params::ParamRealization,time::Real,args...)
  TransientParamRealizationAt(params,Ref(time))
end

function TransientParamRealization(params::ParamRealization,times::AbstractVector{<:Real})
  t0,inner_times... = times
  GenericTransientParamRealization(params,inner_times,t0)
end

get_initial_time(r::GenericTransientParamRealization) = r.t0
get_times(r::GenericTransientParamRealization) = r.times
Base.copy(r::GenericTransientParamRealization) = GenericTransientParamRealization(copy(r.params),copy(r.times),copy(r.t0))
Arrays.testitem(r::GenericTransientParamRealization) = testitem(get_params(r)),r.t0

function Base.getindex(r::GenericTransientParamRealization,i,j)
  TransientParamRealization(
    getindex(get_params(r),i),
    getindex(get_times(r),j),
    r.t0)
end

function Base.iterate(r::GenericTransientParamRealization,state...)
  iterator = Iterators.product(_get_params(r),get_times(r))
  iterate(iterator,state...)
end

get_final_time(r::GenericTransientParamRealization) = last(get_times(r))
get_midpoint_time(r::GenericTransientParamRealization) = (get_final_time(r) + get_initial_time(r)) / 2
get_delta_time(r::GenericTransientParamRealization) = (get_final_time(r) - get_initial_time(r)) / num_times(r)

function change_time!(r::GenericTransientParamRealization{P,T} where P,time::T) where T
  r.times .= time
end

function shift!(r::GenericTransientParamRealization,δ::Real)
  r.times .+= δ
end

function get_at_time(r::GenericTransientParamRealization,time=:initial)
  if time == :initial
    get_at_time(r,get_initial_time(r))
  elseif time == :midpoint
    get_at_time(r,get_midpoint_time(r))
  elseif time == :final
    get_at_time(r,get_final_time(r))
  else
    @notimplemented
  end
end

function get_at_time(r::GenericTransientParamRealization{P,T} where P,time::T)  where T
  TransientParamRealizationAt(get_params(r),Ref(time))
end

struct TransientParamRealizationAt{P,T} <: TransientParamRealization{P,T}
  params::P
  t::Base.RefValue{T}
end

get_initial_time(r::TransientParamRealizationAt) = @notimplemented
get_times(r::TransientParamRealizationAt) = r.t[]
Base.copy(r::TransientParamRealizationAt) = TransientParamRealizationAt(copy(r.params),Ref(copy(r.t)))
Arrays.testitem(r::TransientParamRealizationAt) = testitem(get_params(r)),r.t[]

function Base.getindex(r::TransientParamRealizationAt,i,j)
  @assert j == 1
  new_param = getindex(get_params(r),i)
  TransientParamRealizationAt(new_param,r.t)
end

Base.iterate(r::TransientParamRealizationAt,i...) = iterate(r.params,i...)

function change_time!(r::TransientParamRealizationAt{P,T} where P,time::T) where T
  r.t[] = time
end

function shift!(r::TransientParamRealizationAt,δ::Real)
  r.t[] += δ
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct ParamSpace{P,S} <: AbstractSet{ParamRealization}
  param_domain::P
  sampling_style::S
  function ParamSpace(
    param_domain::P,
    sampling::S=UniformSampling()
    ) where {P<:AbstractVector{<:AbstractVector},S}

    new{P,S}(param_domain,sampling)
  end
end

function Base.show(io::IO,::MIME"text/plain",p::ParamSpace)
  msg = "Set of parameters in $(p.param_domain), sampled with $(p.sampling_style)"
  println(io,msg)
end

function generate_param(p::ParamSpace)
  _value(d,::UniformSampling) = rand(Uniform(first(d),last(d)))
  _value(d,::NormalSampling) = rand(Normal(first(d),last(d)))
  [_value(d,p.sampling_style) for d = p.param_domain]
end

function realization(p::ParamSpace;nparams=1)
  ParamRealization([generate_param(p) for i = 1:nparams])
end

struct TransientParamSpace{P,T} <: AbstractSet{TransientParamRealization}
  parametric_space::P
  temporal_domain::T
end

function TransientParamSpace(
  param_domain::AbstractVector{<:AbstractVector},
  temporal_domain::AbstractVector{<:Real},
  sampling=UniformSampling())

  parametric_space = ParamSpace(param_domain,sampling)
  TransientParamSpace(parametric_space,temporal_domain)
end

function Base.show(io::IO,::MIME"text/plain",p::TransientParamSpace)
  msg = "Set of tuples (p,t) in $(p.parametric_space.param_domain) × $(p.temporal_domain)"
  println(io,msg)
end

function realization(
  p::TransientParamSpace;
  nparams=1,time_locations=eachindex(p.temporal_domain)
  )

  params = realization(p.parametric_space;nparams)
  times = p.temporal_domain[time_locations]
  TransientParamRealization(params,times)
end

function shift!(p::TransientParamSpace,δ::Real)
  p.temporal_domain .+= δ
end

abstract type AbstractParamFunction{P<:ParamRealization} <: Function end

struct ParamFunction{F,P} <: AbstractParamFunction{P}
  fun::F
  params::P
end

const 𝑓ₚ = ParamFunction

function ParamFunction(f::Function,p::AbstractArray)
  @notimplemented "Use a ParamRealization as a parameter input"
end

function ParamFunction(f::Function,r::TrivialParamRealization)
  f(r.params)
end

get_params(f::ParamFunction) = get_params(f.params)
_get_params(f::ParamFunction) = _get_params(f.params)
num_params(f::ParamFunction) = length(_get_params(f))
Base.length(f::ParamFunction) = num_params(f)
Arrays.testitem(f::ParamFunction) = f.fun(testitem(f.params))
Base.getindex(f::ParamFunction,i::Integer) = f.fun(_get_params(f)[i])

function Fields.gradient(f::ParamFunction)
  function _gradient(x,μ)
    gradient(f.fun(μ))(x)
  end
  _gradient(μ) = x -> _gradient(x,μ)
  ParamFunction(_gradient,f.params)
end

function Fields.symmetric_gradient(f::ParamFunction)
  function _symmetric_gradient(x,μ)
    symmetric_gradient(f.fun(μ))(x)
  end
  _symmetric_gradient(μ) = x -> _symmetric_gradient(x,μ)
  ParamFunction(_symmetric_gradient,f.params)
end

function Fields.divergence(f::ParamFunction)
  function _divergence(x,μ)
    divergence(f.fun(μ))(x)
  end
  _divergence(μ) = x -> _divergence(x,μ)
  ParamFunction(_divergence,f.params)
end

function Fields.curl(f::ParamFunction)
  function _curl(x,μ)
    curl(f.fun(μ))(x)
  end
  _curl(μ) = x -> _curl(x,μ)
  ParamFunction(_curl,f.params)
end

function Fields.laplacian(f::ParamFunction)
  function _laplacian(x,μ)
    laplacian(f.fun(μ))(x)
  end
  _laplacian(μ) = x -> _laplacian(x,μ)
  ParamFunction(_laplacian,f.params)
end

# when iterating over a ParamFunction{P}, we return return f(eltype(P)) ∀ index i
function pteval(f::ParamFunction,x)
  T = return_type(f,x)
  v = Vector{T}(undef,length(f))
  @inbounds for (i,μ) in enumerate(get_params(f))
    v[i] = f.fun(x,μ)
  end
  return v
end

Arrays.return_value(f::ParamFunction,x) = f.fun(x,testitem(_get_params(f)))

struct TransientParamFunction{F,P,T} <: AbstractParamFunction{P}
  fun::F
  params::P
  times::T
end

const 𝑓ₚₜ = TransientParamFunction

function TransientParamFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a ParamRealization as a parameter input"
end

function TransientParamFunction(f::Function,r::TrivialParamRealization,t::Real)
  f(r.params,t)
end

function TransientParamFunction(f::Function,r::TransientParamRealization)
  TransientParamFunction(f,get_params(r),get_times(r))
end

get_params(f::TransientParamFunction) = get_params(f.params)
_get_params(f::TransientParamFunction) = _get_params(f.params)
num_params(f::TransientParamFunction) = length(_get_params(f))
get_times(f::TransientParamFunction) = f.times
num_times(f::TransientParamFunction) = length(get_times(f))
Base.length(f::TransientParamFunction) = num_params(f)*num_times(f)
Arrays.testitem(f::TransientParamFunction) = f.fun(testitem(f.params),testitem(f.times))
function Base.getindex(f::TransientParamFunction,i::Integer)
  np = num_params(f)
  p = _get_params(f)[fast_index(i,np)]
  t = get_times(f)[slow_index(i,np)]
  f.fun(p,t)
end

function Fields.gradient(f::TransientParamFunction)
  function _gradient(x,μ,t)
    gradient(f.fun(μ,t))(x)
  end
  _gradient(μ,t) = x -> _gradient(x,μ,t)
  TransientParamFunction(_gradient,f.params,f.times)
end

function Fields.symmetric_gradient(f::TransientParamFunction)
  function _symmetric_gradient(x,μ,t)
    symmetric_gradient(f.fun(μ,t))(x)
  end
  _symmetric_gradient(μ,t) = x -> _symmetric_gradient(x,μ,t)
  TransientParamFunction(_symmetric_gradient,f.params,f.times)
end

function Fields.divergence(f::TransientParamFunction)
  function _divergence(x,μ,t)
    divergence(f.fun(μ,t))(x)
  end
  _divergence(μ,t) = x -> _divergence(x,μ,t)
  TransientParamFunction(_divergence,f.params,f.times)
end

function Fields.curl(f::TransientParamFunction)
  function _curl(x,μ,t)
    curl(f.fun(μ,t))(x)
  end
  _curl(μ,t) = x -> _curl(x,μ,t)
  TransientParamFunction(_curl,f.params,f.times)
end

function Fields.laplacian(f::TransientParamFunction)
  function _laplacian(x,μ,t)
    laplacian(f.fun(μ,t))(x)
  end
  _laplacian(μ,t) = x -> _laplacian(x,μ,t)
  TransientParamFunction(_laplacian,f.params,f.times)
end

function pteval(f::TransientParamFunction,x)
  iterator = Iterators.product(_get_params(f),get_times(f))
  T = return_type(f,x)
  v = Vector{T}(undef,length(f))
  @inbounds for (i,(μ,t)) in enumerate(iterator)
    v[i] = f.fun(x,μ,t)
  end
  return v
end

Arrays.return_value(f::TransientParamFunction,x) = f.fun(x,testitem(_get_params(f)),testitem(get_times(f)))

Arrays.evaluate!(cache,f::AbstractParamFunction,x) = pteval(f,x)

(f::AbstractParamFunction)(x) = evaluate(f,x)

function test_parametric_space()
  α = ParamRealization(rand(10))
  β = ParamRealization([rand(10)])
  @test isa(α,TrivialParamRealization)
  @test isa(α,ParamRealization{Vector{Float64}})
  @test isa(β,ParamRealization{Vector{Vector{Float64}}})
  γ = TransientParamRealization(α,1)
  δ = TransientParamRealization(α,1:10)
  ϵ = TransientParamRealization(β,1:10)
  @test isa(δ,TransientParamRealization{<:TrivialParamRealization,UnitRange{Integer}})
  @test isa(ϵ,TransientParamRealization{ParamRealization{Vector{Vector{Float64}}},UnitRange{Integer}})
  @test length(γ) == 1 && length(δ) == 9 && length(ϵ) == 9
  change_time!(ϵ,11:20)
  @test get_times(get_at_time(ϵ,:final)) == 20
  param_domain = [[1,10],[11,20]]
  p = ParamSpace(param_domain)
  t = 1:10
  pt = TransientParamSpace(param_domain,t)
  μ = realization(p)
  μt = realization(pt)
  @test isa(μ,ParamRealization) && isa(μt,TransientParamRealization)
  a(x,t) = sum(x)*t^2*sin(t)
  a(t) = x -> a(x,t)
  da = ∂t(a)
  aμ(x,μ,t) = sum(μ)*a(x,t)
  aμ(μ,t) = x -> aμ(x,μ,t)
  aμt = 𝑓ₚₜ(aμ,get_params(μt),get_times(μt))
  daμt = ∂t(aμt)
  @test isa(𝑓ₚₜ(a,α,t),Function)
  @test isa(aμt,AbstractParamFunction)
  @test isa(daμt,AbstractParamFunction)
  x = Point(1,2)
  aμtx = aμt(x)
  daμtx = daμt(x)
  for (i,(μ,t)) in enumerate(μt)
    @test aμtx[i] == a(t)(x)*sum(μ)
    @test daμtx[i] == da(t)(x)*sum(μ)
  end
  b(x,μ) = sum(x)*sum(μ)
  b(μ) = x -> b(x,μ)
  bμ = 𝑓ₚ(b,get_params(μ))
  bμx = bμ(x)
  for (i,μ) in enumerate(μ)
    @test b(x,μ) == bμx[i]
  end
  for (i,(μ,t)) in enumerate(μt)
    @test aμ(x,μ,t) == aμtx[i]
  end
  for (i,(μ,t)) in enumerate(μt)
    @test da(x,μ,t) == daμtx[i]
  end
end
