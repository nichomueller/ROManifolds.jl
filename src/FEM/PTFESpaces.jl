"""
A single field FE space with parametric, transient Dirichlet data.
"""
struct PTTrialFESpace{S,B}
  space::S
  dirichlet_μt::Union{Function,Vector{<:Function}}
  Ud0::B

  function PTTrialFESpace(space::S,dirichlet_μt::Union{Function,Vector{<:Function}}) where S
    Ud0 = allocate_trial_space(space)
    B = typeof(Ud0)
    new{S,B}(space,dirichlet_μt,Ud0)
  end
end

"""
Allocate the space to be used as first argument in evaluate!
"""
function allocate_trial_space(U::PTTrialFESpace,args...)
  HomogeneousTrialFESpace(U.space)
end

function allocate_trial_space(U::PTTrialFESpace,μ::Table)
  n = length(μ)
  HomogeneousPTrialFESpace(U.space,n)
end

"""
Peter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function evaluate!(Uμt::T,U::PTTrialFESpace,μ::AbstractVector,t::Real) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = map(o->o(μ,t),U.dirichlet_μt)
  else
    objects_at_μt = U.dirichlet_μt(μ,t)
  end
  TrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

"""
Peter, time evaluation allocating Dirichlet vals
"""
function Arrays.evaluate(U::PTTrialFESpace,params::AbstractVector,t::Real)
  Uμt = allocate_trial_space(U,params)
  evaluate!(Uμt,U,params,t)
  Uμt
end

function evaluate!(Ut::T,U::PTTrialFESpace,params::Table,t::Real) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_t = PTArray(map(o->map(μ->o(μ,t),params),U.dirichlet_μt))
  else
    objects_at_t = PTArray(map(μ->U.dirichlet_μt(μ,t),params))
  end
  PTrialFESpace!(Ut,objects_at_t)
  Ut
end

function Arrays.evaluate(U::PTTrialFESpace,params::AbstractVector,t::Real)
  Ut = allocate_trial_space(U,params)
  evaluate!(Ut,U,params,t)
  Ut
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Arrays.evaluate(U::PTTrialFESpace,::Nothing,::Nothing) = U.Ud0
Arrays.evaluate(U::PTTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::SingleFieldFESpace)(μ,t) = U
(U::PTTrialFESpace)(μ,t) = evaluate(U,μ,t)
(U::PTTrialFESpace)(μ) = evaluate(U,μ)

"""
Time derivative of the Dirichlet functions
"""
∂ₚt(U::PTTrialFESpace) =
  PTTrialFESpace(U.space,∂ₚt.(U.dirichlet_μt))

"""
Time 2nd derivative of the Dirichlet functions
"""
∂ₚtt(U::PTTrialFESpace) =
  PTTrialFESpace(U.space,∂ₚtt.(U.dirichlet_μt))

# Define the PTrialFESpace interface for affine spaces

function Arrays.evaluate!(::FESpace,U::FESpace,::AbstractVector,::Real)
  U
end

function Arrays.evaluate(U::FESpace,::AbstractVector,::Real)
  U
end

function Arrays.evaluate(U::FESpace,::Nothing,::Nothing)
  U
end

# Define the interface for MultiField

struct PTMultiFieldTrialFESpace
  spaces::Vector
end
Base.iterate(m::PTMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::PTMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::PTMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::PTMultiFieldTrialFESpace) = length(m.spaces)

function allocate_trial_space(U::PTMultiFieldTrialFESpace,args...)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function allocate_trial_space(U::PTMultiFieldTrialFESpace,μ::Table)
  n = length(μ)*length(t)
  spaces = map(fe->HomogeneousPTrialFESpace(x,n),U.spaces)
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμt::T,U::PTMultiFieldTrialFESpace,μ,t::Real) where T
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Arrays.evaluate(U::PTMultiFieldTrialFESpace,μ,t::Real)
  Uμt = allocate_trial_space(U,μ)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Arrays.evaluate(U::PTMultiFieldTrialFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([fesp(nothing,nothing) for fesp in U.spaces])
end

(U::PTMultiFieldTrialFESpace)(μ,t) = evaluate(U,μ,t)
(U::PTMultiFieldTrialFESpace)(μ) = evaluate(U,μ)

function ∂ₚt(U::PTMultiFieldTrialFESpace)
  spaces = ∂ₚt.(U.spaces)
  PTMultiFieldFESpace(spaces)
end

function PTMultiFieldFESpace(spaces::Vector)
  PTMultiFieldTrialFESpace(spaces)
end

function PTMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end
