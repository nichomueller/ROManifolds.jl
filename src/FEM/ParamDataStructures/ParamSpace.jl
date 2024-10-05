"""
    AbstractRealization

Type representing parametric realizations, i.e. samples extracted from a given
parameter space. Two categories of such realizations are implemented:
- [`Realization`](@ref).
- [`TransientRealization`](@ref).

"""

abstract type AbstractRealization end

"""
    Realization{P<:AbstractVector} <: AbstractRealization

Represents standard parametric realizations, i.e. samples extracted from
a given parameter space. The field `params` is most commonly a vector of vectors.
When the parameters are scalars, they still need to be defined as vectors of
vectors of unit length. In other words, we treat the case in which `params` is a
vector of numbers as the case in which `params` is a vector of one vector.

# Examples

```jldoctest
julia> r = Realization([[1, 2],[3, 4]])
Realization{Vector{Vector{Int64}}}([[1, 2], [3, 4]])
julia> μ = r[1]
Realization{Vector{Int64}}([1, 2])
julia> r′ = Realization([1, 2])
Realization{Vector{Int64}}([1, 2])
julia> μ′ = r′[1]
Realization{Vector{Int64}}([1, 2])
```

"""

struct Realization{P<:AbstractVector} <: AbstractRealization
  params::P
end

const TrivialRealization = Realization{<:AbstractVector{<:Real}}

get_params(r::Realization) = r # we only want to deal with a Realization type
_get_params(r::Realization) = r.params # this function should stay local
_get_params(r::TrivialRealization) = [r.params] # this function should stay local
num_params(r::Realization) = length(_get_params(r))
Base.length(r::Realization) = num_params(r)
Base.getindex(r::Realization,i) = Realization(getindex(_get_params(r),i))
Base.copy(r::Realization) = Realization(copy(_get_params(r)))
Arrays.testitem(r::Realization) = testitem(_get_params(r))

# when iterating over a Realization{P}, we return eltype(P) ∀ index i
function Base.iterate(r::Realization,state=1)
  if state > length(r)
    return nothing
  end
  rstate = _get_params(r)[state]
  return rstate, state+1
end

function Base.zero(r::Realization)
  μ1 = first(_get_params(r))
  Realization(zeros(eltype(μ1),length(μ1)) .+ 1e-16)
end

function mean(r::Realization)
  Realization(mean(_get_params(r)))
end

"""
    TransientRealization{P<:Realization,T<:Real} <: AbstractRealization

Represents temporal parametric realizations, i.e. samples extracted from
a given parameter space for every time step in a temporal range. The most obvious
application of this type are transient PDEs, where an initial condition is given.
Following this convention, the initial time instant is kept separate from the
other time steps.

"""

abstract type TransientRealization{P<:Realization,T<:Real} <: AbstractRealization end

Base.length(r::TransientRealization) = num_params(r)*num_times(r)
get_params(r::TransientRealization) = get_params(r.params)
_get_params(r::TransientRealization) = _get_params(r.params)
num_params(r::TransientRealization) = num_params(r.params)
num_times(r::TransientRealization) = length(get_times(r))

"""
    GenericTransientRealization{P,T,A} <: TransientRealization{P,T}

Most standard implementation of an transient parametric realization.

"""

struct GenericTransientRealization{P,T,A} <: TransientRealization{P,T}
  params::P
  times::A
  t0::T
end

function TransientRealization(params::Realization,times::AbstractVector{<:Real},t0::Real)
  GenericTransientRealization(params,times,t0)
end

function TransientRealization(params::Realization,time::Real,args...)
  TransientRealizationAt(params,Ref(time))
end

function TransientRealization(params::Realization,times::AbstractVector{<:Real})
  t0,inner_times... = times
  GenericTransientRealization(params,inner_times,t0)
end

get_initial_time(r::GenericTransientRealization) = r.t0
get_times(r::GenericTransientRealization) = r.times
Base.copy(r::GenericTransientRealization) = GenericTransientRealization(copy(r.params),copy(r.times),copy(r.t0))
Arrays.testitem(r::GenericTransientRealization) = testitem(get_params(r)),r.t0

function Base.getindex(r::GenericTransientRealization,i,j)
  TransientRealization(
    getindex(get_params(r),i),
    getindex(get_times(r),j),
    r.t0)
end

function Base.iterate(r::GenericTransientRealization,state...)
  iterator = Iterators.product(_get_params(r),get_times(r))
  iterate(iterator,state...)
end

function Base.zero(r::GenericTransientRealization)
  GenericTransientRealization(zero(get_params(r)),get_times(r),get_initial_time(r))
end

function mean(r::GenericTransientRealization)
  GenericTransientRealization(mean(get_params(r)),get_times(r),get_initial_time(r))
end

get_final_time(r::GenericTransientRealization) = last(get_times(r))
get_midpoint_time(r::GenericTransientRealization) = (get_final_time(r) + get_initial_time(r)) / 2
get_delta_time(r::GenericTransientRealization) = (get_final_time(r) - get_initial_time(r)) / num_times(r)

function change_time!(r::GenericTransientRealization{P,T} where P,time::T) where T
  r.times .= time
end

function shift!(r::GenericTransientRealization,δ::Real)
  r.times .+= δ
end

function get_at_time(r::GenericTransientRealization,time=:initial)
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

function get_at_time(r::GenericTransientRealization{P,T} where P,time::T)  where T
  TransientRealizationAt(get_params(r),Ref(time))
end

"""
    TransientRealizationAt{P,T} <: TransientRealization{P,T}

Represents a GenericTransientRealization{P,T} at a certain time instant `t`.
For reusability purposes, the time instant `t` is stored as a Base.RefValue{T}.

"""

struct TransientRealizationAt{P,T} <: TransientRealization{P,T}
  params::P
  t::Base.RefValue{T}
end

get_initial_time(r::TransientRealizationAt) = @notimplemented
get_times(r::TransientRealizationAt) = r.t[]
Base.copy(r::TransientRealizationAt) = TransientRealizationAt(copy(r.params),Ref(copy(r.t)))
Arrays.testitem(r::TransientRealizationAt) = testitem(get_params(r)),r.t[]

function Base.getindex(r::TransientRealizationAt,i,j)
  @assert j == 1
  new_param = getindex(get_params(r),i)
  TransientRealizationAt(new_param,r.t)
end

Base.iterate(r::TransientRealizationAt,i...) = iterate(r.params,i...)

function change_time!(r::TransientRealizationAt{P,T} where P,time::T) where T
  r.t[] = time
end

function shift!(r::TransientRealizationAt,δ::Real)
  r.t[] += δ
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

"""
    ParamSpace{P,S} <: AbstractSet{Realization}

Represents a standard set of parameters.

"""

struct ParamSpace{P,S} <: AbstractSet{Realization}
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

"""
    realization(p::ParamSpace;nparams=1) -> Realization
    realization(p::TransientParamSpace;nparams=1) -> TransientRealization

Extraction of a set of parameters from a given parametric space
"""

function realization(p::ParamSpace;nparams=1)
  Realization([generate_param(p) for i = 1:nparams])
end

"""
    TransientParamSpace{P,T} <: AbstractSet{TransientRealization}

Represents a transient set of parameters.

"""

struct TransientParamSpace{P,T} <: AbstractSet{TransientRealization}
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
  TransientRealization(params,times)
end

function shift!(p::TransientParamSpace,δ::Real)
  p.temporal_domain .+= δ
end

"""
    AbstractParamFunction{P<:Realization} <: Function

Representation of parametric functions with domain a parametric space.
Two categories of such functions are implemented:
- [`ParamFunction`](@ref).
- [`TransientParamFunction`](@ref).

"""

abstract type AbstractParamFunction{P<:Realization} <: Function end

"""
    ParamFunction{F,P} <: AbstractParamFunction{P}

Representation of parametric functions with domain a parametric space. Given a
function `f` : Ω₁ × ... × Ωₙ × U+1D4DF, where U+1D4DF is a `ParamSpace`,
the evaluation of `f` in `μ ∈ U+1D4DF` returns the restriction of `f` to Ω₁ × ... × Ωₙ

# Examples

```jldoctest
julia> U+1D4DF = ParamSpace([[0, 1],[0, 1]])
Set of parameters in [[0, 1], [0, 1]], sampled with UniformSampling()
julia> μ = realization(U+1D4DF; nparams = 2)
Realization{Vector{Vector{Float64}}}([...])
julia> a(x, μ) = sum(x) * sum(μ)
a (generic function with 1 method)
julia> a(μ) = x -> a(x, μ)
a (generic function with 2 methods)
julia> aμ = ParamFunction(a, μ)
#15 (generic function with 1 method)
julia> aμ(Point(0, 1))
2-element Vector{Float64}:
 0.068032791851195
 0.9263487710801264
```

"""

struct ParamFunction{F,P} <: AbstractParamFunction{P}
  fun::F
  params::P
end

const 𝑓ₚ = ParamFunction

function ParamFunction(f::Function,p::AbstractArray)
  @notimplemented "Use a Realization as a parameter input"
end

function ParamFunction(f::Function,r::TrivialRealization)
  f(r.params)
end

get_params(f::ParamFunction) = get_params(f.params)
_get_params(f::ParamFunction) = _get_params(f.params)
num_params(f::ParamFunction) = length(_get_params(f))
Base.length(f::ParamFunction) = num_params(f)
Arrays.testitem(f::ParamFunction) = f.fun(testitem(f.params))
Base.getindex(f::ParamFunction,i::Integer) = f.fun(_get_params(f)[i])

function Base.:*(f::ParamFunction,α::Number)
  _fun(x,μ) = α*f.fun(x,μ)
  _fun(μ) = x -> _fun(x,μ)
  ParamFunction(_fun,f.params)
end

Base.:*(α::Number,f::ParamFunction) = f*α

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

"""
    TransientParamFunction{F,P,T} <: AbstractParamFunction{P}

Representation of parametric functions with domain a transient parametric space.
Given a function `f` : Ω₁ × ... × Ωₙ × [t₁,t₂] × U+1D4DF, where [t₁,t₂] is a
temporal domain and U+1D4DF is a `ParamSpace`, or equivalently
`f` : Ω₁ × ... × Ωₙ × [t₁,t₂] × U+1D4E3 U+1D4DF, where U+1D4E3 U+1D4DF is a
`TransientParamSpace`, the evaluation of `f` in `μ ∈ U+1D4E3 U+1D4DF` returns
the restriction of `f` to Ω₁ × ... × Ωₙ

"""

struct TransientParamFunction{F,P,T} <: AbstractParamFunction{P}
  fun::F
  params::P
  times::T
end

const 𝑓ₚₜ = TransientParamFunction

function TransientParamFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a Realization as a parameter input"
end

function TransientParamFunction(f::Function,r::TrivialRealization,t::Real)
  f(r.params,t)
end

function TransientParamFunction(f::Function,r::TransientRealization)
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

function Base.:*(f::TransientParamFunction,α::Number)
  _fun(x,μ,t) = α*f.fun(x,μ,t)
  _fun(μ,t) = x -> _fun(x,μ,t)
  TransientParamFunction(_fun,f.params,f.times)
end

Base.:*(α::Number,f::TransientParamFunction) = f*α

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
  α = Realization(rand(10))
  β = Realization([rand(10)])
  @test isa(α,TrivialRealization)
  @test isa(α,Realization{Vector{Float64}})
  @test isa(β,Realization{Vector{Vector{Float64}}})
  γ = TransientRealization(α,1)
  δ = TransientRealization(α,1:10)
  ϵ = TransientRealization(β,1:10)
  @test isa(δ,TransientRealization{<:TrivialRealization,UnitRange{Integer}})
  @test isa(ϵ,TransientRealization{Realization{Vector{Vector{Float64}}},UnitRange{Integer}})
  @test length(γ) == 1 && length(δ) == 9 && length(ϵ) == 9
  change_time!(ϵ,11:20)
  @test get_times(get_at_time(ϵ,:final)) == 20
  param_domain = [[1,10],[11,20]]
  p = ParamSpace(param_domain)
  t = 1:10
  pt = TransientParamSpace(param_domain,t)
  μ = realization(p)
  μt = realization(pt)
  @test isa(μ,Realization) && isa(μt,TransientRealization)
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
