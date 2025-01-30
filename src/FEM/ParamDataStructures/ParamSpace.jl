"""
    abstract type AbstractRealization end

Type representing parametric realizations, i.e. samples extracted from a given
parameter space. Two categories of such realizations are implemented:
- `Realization`.
- `TransientRealization`.
"""
abstract type AbstractRealization end

"""
    get_params(r::AbstractRealization) -> Realization
"""
get_params(r::AbstractRealization) = @abstractmethod

# this function should stay local
_get_params(r::AbstractRealization) = @abstractmethod

"""
    num_params(r::AbstractRealization) -> Int
"""
num_params(r::AbstractRealization) = length(_get_params(r))

param_length(r::AbstractRealization) = length(r)

param_getindex(r::AbstractRealization,i::Integer) = getindex(r,i)

"""
    struct Realization{P<:AbstractVector} <: AbstractRealization
      params::P
    end

Represents standard parametric realizations, i.e. samples extracted from
a given parameter space. The field `params` is most commonly a vector of vectors.
When the parameters are scalars, they still need to be defined as vectors of
vectors of unit length. In other words, we treat the case in which `params` is a
vector of numbers as the case in which `params` is a vector of one vector.
"""
struct Realization{P<:AbstractVector} <: AbstractRealization
  params::P
end

const TrivialRealization = Realization{<:AbstractVector{<:Real}}

get_params(r::Realization) = r # we only want to deal with a Realization type
_get_params(r::Realization) = r.params # this function should stay local
_get_params(r::TrivialRealization) = [r.params] # this function should stay local

Base.length(r::Realization) = num_params(r)

Base.getindex(r::Realization,i) = Realization(getindex(_get_params(r),i))

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

"""
    abstract type TransientRealization{P<:Realization,T<:Real} <: AbstractRealization end

Represents temporal parametric realizations, i.e. samples extracted from
a given parameter space for every time step in a temporal range. The most obvious
application of this type are transient PDEs, where an initial condition is given.
Following this convention, the initial time instant is kept separate from the
other time steps.
"""
abstract type TransientRealization{P<:Realization,T<:Real} <: AbstractRealization end

"""
    get_times(r::TransientRealization) -> Any
"""
get_times(r::TransientRealization) = @abstractmethod

"""
    get_times(r::TransientRealization) -> Int
"""
num_times(r::TransientRealization) = length(get_times(r))

Base.length(r::TransientRealization) = num_params(r)*num_times(r)

"""
    struct GenericTransientRealization{P,T,A} <: TransientRealization{P,T}
      params::P
      times::A
      t0::T
    end

Most standard implementation of a `TransientRealization`.
"""
struct GenericTransientRealization{P,T,A} <: TransientRealization{P,T}
  params::P
  times::A
  t0::T
end

get_params(r::GenericTransientRealization) = get_params(r.params)
_get_params(r::GenericTransientRealization) = _get_params(r.params)
get_times(r::GenericTransientRealization) = r.times

function TransientRealization(params::Realization,times::AbstractVector{<:Real},t0::Real)
  GenericTransientRealization(params,times,t0)
end

function TransientRealization(params::Realization,time::Real,args...)
  TransientRealization(params,[time],args...)
end

function TransientRealization(params::Realization,times::AbstractVector{<:Real})
  t0,inner_times... = times
  TransientRealization(params,inner_times,t0)
end

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

"""
    get_initial_time(r::GenericTransientRealization) -> Real
"""
get_initial_time(r::GenericTransientRealization) = r.t0

"""
    get_final_time(r::GenericTransientRealization) -> Real
"""
get_final_time(r::GenericTransientRealization) = last(get_times(r))

get_midpoint_time(r::GenericTransientRealization) = (get_final_time(r) + get_initial_time(r)) / 2

get_delta_time(r::GenericTransientRealization) = (get_final_time(r) - get_initial_time(r)) / num_times(r)

function change_time!(r::GenericTransientRealization{P,T} where P,time::T) where T
  r.times .= time
end

"""
    shift!(r::TransientRealization,δ::Real) -> Nothing

In-place uniform shifting by a constant `δ` of the temporal domain in the
realization `r`
"""
function shift!(r::GenericTransientRealization,δ::Real)
  r.times .+= δ
end

"""
    get_at_time(r::GenericTransientRealization,time) -> TransientRealizationAt

Returns a [`TransientRealizationAt`](@ref) from a [`GenericTransientRealization`](@ref)
at a time instant specified by `time`
"""
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
    struct TransientRealizationAt{P,T} <: TransientRealization{P,T}
      params::P
      t::Base.RefValue{T}
    end

Represents a GenericTransientRealization{P,T} at a certain time instant `t`.
To avoid making it a mutable struct, the time instant `t` is stored as a Base.RefValue{T}.
"""
struct TransientRealizationAt{P,T} <: TransientRealization{P,T}
  params::P
  t::Base.RefValue{T}
end

get_params(r::TransientRealizationAt) = get_params(r.params)

_get_params(r::TransientRealizationAt) = _get_params(r.params)

get_times(r::TransientRealizationAt) = r.t[]

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

"""
    abstract type SamplingStyle end

Subtypes:
- [`UniformSampling`](@ref)
- [`NormalSampling`](@ref)
- [`HaltonSampling`](@ref)
"""
abstract type SamplingStyle end

"""
"""
struct UniformSampling <: SamplingStyle end

function generate_param(::UniformSampling,param_domain)
  [rand(Uniform(first(d),last(d))) for d = param_domain]
end

"""
"""
struct NormalSampling <: SamplingStyle end

function generate_param(::NormalSampling,param_domain)
  [rand(Uniform(first(d),last(d))) for d = param_domain]
end

"""
"""
struct HaltonSampling <: SamplingStyle end

function _generate_params(::HaltonSampling,param_domain,nparams;start=1,kwargs...)
  d = length(param_domain)
  hs = HaltonPoint(d;length=nparams,start=start)
  hs′ = collect(hs)
  for x in hs′
    for (di,xdi) in enumerate(x)
      a,b = param_domain[di]
      x[di] = a + (b-a)*xdi
    end
  end
  return hs′
end

"""
    struct ParamSpace{P<:AbstractVector{<:AbstractVector},S<:SamplingStyle} <: AbstractSet{Realization}
      param_domain::P
      sampling_style::S
    end

Fields:
- `param_domain`: domain of definition of the parameters
- `sampling_style`: distribution on `param_domain` according to which we can
  sample the parameters (by default it is set to `HaltonSampling`)
"""
struct ParamSpace{P<:AbstractVector{<:AbstractVector},S<:SamplingStyle} <: AbstractSet{Realization}
  param_domain::P
  sampling_style::S
end

get_sampling_style(p::ParamSpace) = p.sampling_style

function ParamSpace(param_domain::AbstractVector{<:AbstractVector})
  sampling_style = HaltonSampling()
  ParamSpace(param_domain,sampling_style)
end

function ParamSpace(domain_tuple::NTuple{N,T},args...) where {N,T}
  @notimplementedif !isconcretetype(T)
  @notimplementedif isodd(N)
  param_domain = Vector{Vector{T}}(undef,Int(N/2))
  for (i,n) in enumerate(1:2:N)
    param_domain[i] = [domain_tuple[n],domain_tuple[n+1]]
  end
  ParamSpace(param_domain,args...)
end

function Base.show(io::IO,::MIME"text/plain",p::ParamSpace)
  msg = "Set of parameters in $(p.param_domain), sampled with $(p.sampling_style)"
  println(io,msg)
end

function _generate_params(sampling::SamplingStyle,param_domain,nparams;kwargs...)
  [generate_param(sampling,param_domain;kwargs...) for i = 1:nparams]
end

function _generate_params(sampling::Symbol,args...;kwargs...)
  if sampling == :uniform
    _generate_params(UniformSampling(),args...;kwargs...)
  elseif sampling == :normal
    _generate_params(NormalSampling(),args...;kwargs...)
  elseif sampling == :halton
    _generate_params(HaltonSampling(),args...;kwargs...)
  else
    @notimplemented "Need to implement more sampling strategies"
  end
end

"""
    realization(p::ParamSpace;nparams=1,sampling=get_sampling_style(p),kwargs...) -> Realization
    realization(p::TransientParamSpace;nparams=1,sampling=get_sampling_style(p),kwargs...) -> TransientRealization

Extraction of a set of `nparams` parameters from a given parametric space, by
default according to the sampling strategy specified in `p`.
"""
function realization(p::ParamSpace;nparams=1,sampling=get_sampling_style(p),kwargs...)
  params = _generate_params(sampling,p.param_domain,nparams;kwargs...)
  Realization(params)
end

"""
    struct TransientParamSpace{P<:ParamSpace,T} <: AbstractSet{TransientRealization}
      parametric_space::P
      temporal_domain::T
    end

Fields:
- `parametric_space`: underlying parameter space
- `temporal_domain`: underlying temporal space

It represents, in essence, the set of tuples (p,t) in `parametric_space` × `temporal_domain`
"""
struct TransientParamSpace{P<:ParamSpace,T} <: AbstractSet{TransientRealization}
  parametric_space::P
  temporal_domain::T
end

function TransientParamSpace(
  param_domain::Union{Tuple,AbstractVector},
  temporal_domain::AbstractVector{<:Real},
  args...)

  parametric_space = ParamSpace(param_domain,args...)
  TransientParamSpace(parametric_space,temporal_domain)
end

function Base.show(io::IO,::MIME"text/plain",p::TransientParamSpace)
  msg = "Set of tuples (p,t) in $(p.parametric_space.param_domain) × $(p.temporal_domain)"
  println(io,msg)
end

function realization(
  p::TransientParamSpace;
  time_locations=eachindex(p.temporal_domain),
  kwargs...
  )

  params = realization(p.parametric_space;kwargs...)
  times = p.temporal_domain[time_locations]
  TransientRealization(params,times)
end

function shift!(p::TransientParamSpace,δ::Real)
  p.temporal_domain .+= δ
end

"""
    abstract type AbstractParamFunction{P<:Realization} <: Function end

Representation of parametric functions with domain a parametric space.
Subtypes:
- [`ParamFunction`](@ref)
- [`TransientParamFunction`](@ref)
"""
abstract type AbstractParamFunction{P<:Realization} <: Function end

param_length(f::AbstractParamFunction) = length(f)
param_getindex(f::AbstractParamFunction,i::Integer) = getindex(f,i)
Arrays.testitem(f::AbstractParamFunction) = param_getindex(f,1)
Arrays.evaluate!(cache,f::AbstractParamFunction,x) = pteval(f,x)
Arrays.evaluate!(cache,f::AbstractParamFunction,x::CellPoint) = CellField(f,get_triangulation(x))(x)
(f::AbstractParamFunction)(x) = evaluate(f,x)

function Arrays.return_cache(f::AbstractParamFunction,x)
  v = return_value(f,x)
  V = eltype(v)
  plength = param_length(f)
  cache = Vector{V}(undef,plength)
  return cache
end

"""
    struct ParamFunction{F,P} <: AbstractParamFunction{P}
      fun::F
      params::P
    end

Representation of parametric functions with domain a parametric space. Given a
function `f` : Ω₁ × ... × Ωₙ × U+1D4DF, where U+1D4DF is a `ParamSpace`,
the evaluation of `f` in `μ ∈ U+1D4DF` returns the restriction of `f` to Ω₁ × ... × Ωₙ
"""
struct ParamFunction{F,P} <: AbstractParamFunction{P}
  fun::F
  params::P
end

function ParamFunction(f::Function,p::Union{AbstractArray,TransientRealization})
  @notimplemented "Use a Realization as a parameter input"
end

function ParamFunction(f::Function,pt::TransientRealizationAt)
  ParamFunction(f,get_params(pt))
end

get_params(f::ParamFunction) = get_params(f.params)
_get_params(f::ParamFunction) = _get_params(f.params)
num_params(f::ParamFunction) = length(_get_params(f))
Base.length(f::ParamFunction) = num_params(f)
Base.getindex(f::ParamFunction,i::Integer) = f.fun(_get_params(f)[i])

function Base.:*(f::ParamFunction,α::Number)
  _fun(μ) = x -> α*f.fun(μ)(x)
  ParamFunction(_fun,f.params)
end

Base.:*(α::Number,f::ParamFunction) = f*α

for op in (:(Fields.gradient),:(Fields.symmetric_gradient),:(Fields.divergence),
  :(Fields.curl),:(Fields.laplacian))
  @eval begin
    function ($op)(f::ParamFunction)
      _op(μ) = x -> $op(f.fun(μ))(x)
      ParamFunction(_op,f.params)
    end
  end
end

# when iterating over a ParamFunction{P}, we return return f(eltype(P)) ∀ index i
function pteval(f::ParamFunction,x)
  T = return_type(f,x)
  v = Vector{T}(undef,length(f))
  @inbounds for (i,μ) in enumerate(get_params(f))
    v[i] = f.fun(μ)(x)
  end
  return v
end

Arrays.return_value(f::ParamFunction,x) = f.fun(testitem(_get_params(f)))(x)

"""
    struct TransientParamFunction{F,P,T} <: AbstractParamFunction{P}
      fun::F
      params::P
      times::T
    end

Representation of parametric functions with domain a transient parametric space.
Given a function `f` : Ω₁ × ... × Ωₙ × U+1D4DF × [t₁,t₂], where [t₁,t₂] is a
temporal domain and U+1D4DF is a `ParamSpace`, or equivalently
`f` : Ω₁ × ... × Ωₙ × U+1D4E3 U+1D4DF × [t₁,t₂], where U+1D4E3 U+1D4DF is a
`TransientParamSpace`, the evaluation of `f` in `(μ,t) ∈ U+1D4E3 U+1D4DF × [t₁,t₂]`
returns the restriction of `f` to Ω₁ × ... × Ωₙ
"""
struct TransientParamFunction{F,P,T} <: AbstractParamFunction{P}
  fun::F
  params::P
  times::T
end

function TransientParamFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a TransientRealization as a parameter input"
end

function TransientParamFunction(f::Function,r::Realization)
  @notimplemented "Use a TransientRealization as a parameter input"
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
function Base.getindex(f::TransientParamFunction,i::Integer)
  np = num_params(f)
  p = _get_params(f)[fast_index(i,np)]
  t = get_times(f)[slow_index(i,np)]
  f.fun(p,t)
end

function Base.:*(f::TransientParamFunction,α::Number)
  _fun(μ,t) = x -> α*f.fun(μ,t)(x)
  TransientParamFunction(_fun,f.params,f.times)
end

Base.:*(α::Number,f::TransientParamFunction) = f*α

for op in (:(Fields.gradient),:(Fields.symmetric_gradient),:(Fields.divergence),
  :(Fields.curl),:(Fields.laplacian))
  @eval begin
    function ($op)(f::TransientParamFunction)
      _op(μ,t) = x -> $op(f.fun(μ,t))(x)
      TransientParamFunction(_op,f.params,f.times)
    end
  end
end

function pteval(f::TransientParamFunction,x)
  iterator = Iterators.product(_get_params(f),get_times(f))
  T = return_type(f,x)
  v = Vector{T}(undef,length(f))
  @inbounds for (i,(μ,t)) in enumerate(iterator)
    v[i] = f.fun(μ,t)(x)
  end
  return v
end

Arrays.return_value(f::TransientParamFunction,x) = f.fun(testitem(_get_params(f)),testitem(get_times(f)))(x)
