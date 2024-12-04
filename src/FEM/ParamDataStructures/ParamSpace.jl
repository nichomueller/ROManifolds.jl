"""
    AbstractRealization

Type representing parametric realizations, i.e. samples extracted from a given
parameter space. Two categories of such realizations are implemented:
- [`Realization`](@ref).
- [`TransientRealization`](@ref).

"""

abstract type AbstractRealization end

param_length(r::AbstractRealization) = length(r)

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
  TransientRealization(params,[time],args...)
end

function TransientRealization(params::Realization,times::AbstractVector{<:Real})
  t0,inner_times... = times
  TransientRealization(params,inner_times,t0)
end

get_initial_time(r::GenericTransientRealization) = r.t0
get_times(r::GenericTransientRealization) = r.times

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
struct HaltonSampling <: SamplingStyle end
struct SmolyakSampling <: SamplingStyle end

"""
    ParamSpace{P,S} <: AbstractSet{Realization}

Represents a standard set of parameters.

"""

struct ParamSpace{P<:AbstractVector{<:AbstractVector},S<:SamplingStyle} <: AbstractSet{Realization}
  param_domain::P
  sampling_style::S
end

dimension(p::ParamSpace) = length(p.param_domain)

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

function _generate_param(p::ParamSpace{P,UniformSampling} where P)
  [rand(Uniform(first(d),last(d))) for d = p.param_domain]
end

function _generate_param(p::ParamSpace{P,NormalSampling} where P)
  [rand(Normal(first(d),last(d))) for d = p.param_domain]
end

"""
    realization(p::ParamSpace;nparams=1) -> Realization
    realization(p::TransientParamSpace;nparams=1) -> TransientRealization

Extraction of a set of parameters from a given parametric space

"""
function realization(p::ParamSpace{P,S} where {P,S};nparams=1,kwargs...)
  Realization([_generate_param(p) for i = 1:nparams])
end

function realization(p::ParamSpace{P,HaltonSampling} where P;nparams=1,random=false,kwargs...)
  if random
    p′ = ParamSpace(p.param_domain,UniformSampling())
    realization(p′;nparams)
  else
    hs = shifted_halton(p,nparams)
    Realization(hs)
  end
end

function realization(
  p::ParamSpace{P,SmolyakSampling} where P;
  level=dimension(p),
  grid=smolyak_grid(p),
  nparams=length(grid),
  random=false)

  if random
    p′ = ParamSpace(p.param_domain,UniformSampling())
    realization(p′;nparams)
  else
    if nparams > length(grid)
      realization(p;level=level+1,nparams=nparams)
    else
      Realization(grid[1:nparams])
    end
  end
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

get_params(f::ParamFunction) = get_params(f.params)
_get_params(f::ParamFunction) = _get_params(f.params)
num_params(f::ParamFunction) = length(_get_params(f))
Base.length(f::ParamFunction) = num_params(f)
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

Arrays.evaluate!(cache,f::AbstractParamFunction,x::CellPoint) = CellField(f,get_triangulation(x))(x)

(f::AbstractParamFunction)(x) = evaluate(f,x)

# Halton utils

function shifted_halton(p::ParamSpace,nparams::Integer)
  domain = p.param_domain
  d = dimension(p)
  hs = HaltonPoint(d,length=nparams)
  hs′ = collect(hs)
  for x in hs′
    for (di,xdi) in enumerate(x)
      a,b = domain[di]
      x[di] = a + (b-a)*xdi
    end
  end
  return hs′
end

# Smolyak utils

function SmolyakApprox.scale_nodes!(
  grid::AbstractVector{<:AbstractVector},
  domain::AbstractVector{<:AbstractVector})

  @inbounds for (i,gridi) in enumerate(grid)
    gridi = grid[i]
    for (j,gridij) in enumerate(gridi)
      gridi[j] = domain[j][2] + (1.0+gridij)*(domain[j][1]-domain[j][2])*0.5
    end
  end

end

function uniform_extrema(N::S,domain=[1.0,-1.0]) where {S<:Integer}
  points = fill((domain[1]+domain[2])/2,N)
  L = domain[2]-domain[1]
  @inbounds for i = 1:div(N,2)
    x = L/(2*i)
    points[i]     += x
    points[N-i+1] -= x
  end
  return points
end

function SmolyakApprox.smolyak_grid(
  p::ParamSpace,
  node_type=SmolyakApprox.chebyshev_extrema,
  level=dimension(p))

  d = dimension(p)
  domain = p.param_domain

  T = Vector{Float64}

  multi_index = SmolyakApprox.generate_multi_index(d,level)
  unique_multi_index = sort(unique(multi_index))
  unique_node_number = SmolyakApprox.m_i.(unique_multi_index)

  # Create base nodes to be used in the sparse grid

  base_nodes = Vector{T}(undef,length(unique_node_number))
  for i in eachindex(unique_node_number)
    base_nodes[i] = node_type(unique_node_number[i])
  end

  # Determine the unique nodes introduced at each higher level

  unique_base_nodes = Vector{T}(undef,length(unique_node_number))
  unique_base_nodes[1] = base_nodes[1]
  for i = 2:length(unique_base_nodes)
    unique_base_nodes[i] = setdiff(base_nodes[i],base_nodes[i-1])
  end

  # Construct the sparse grid from the unique nodes

  grid = Vector{T}(undef,SmolyakApprox.determine_grid_size(multi_index)[1])
  l = 1
  @inbounds for j in axes(multi_index,1)
    new_nodes = unique_base_nodes[multi_index[j,1]] # Here new_nodes is a 1d array
    for i = 2:d
      new_nodes = SmolyakApprox.combine_nodes(new_nodes,unique_base_nodes[multi_index[j,i]])  # Here new_nodes becomes a 2d array
    end
    m = size(new_nodes,1)
    for (ik,k) in enumerate(l:l+m-1)
      grid[k] = new_nodes[ik,:]
    end
    l += m
  end

  # Now scale the nodes to the desired domain

  SmolyakApprox.scale_nodes!(grid,domain)

  return grid
end
