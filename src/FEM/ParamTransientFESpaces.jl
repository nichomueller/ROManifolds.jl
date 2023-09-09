"""
A single field FE space with parametric, transient Dirichlet data.
"""
struct ParamTransientTrialFESpace{S,B}
  space::S
  dirichlet_μt::Union{Function,Vector{<:Function}}
  Ud0::B

  function ParamTransientTrialFESpace(space::S,dirichlet_μt::Union{Function,Vector{<:Function}}) where S
    Ud0 = HomogeneousTrialFESpace(space)
    B = typeof(Ud0)
    new{S,B}(space,dirichlet_μt,Ud0)
  end
end

function ParamTransientTrialFESpace(space::S) where S
  HomogeneousTrialFESpace(space)
end

"""
Allocate the space to be used as first argument in evaluate!
"""
function allocate_trial_space(U::ParamTransientTrialFESpace,args...)
  HomogeneousTrialFESpace(U.space,args...)
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::AbstractVector,t::Real) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = map(o->o(μ,t),U.dirichlet_μt)
  else
    objects_at_μt = U.dirichlet_μt(μ,t)
  end
  TrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

"""
Parameter, time evaluation allocating Dirichlet vals
"""
function Arrays.evaluate(U::ParamTransientTrialFESpace,μ::AbstractVector,t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

# Interface for time marching across parameters
function FESpaces.HomogeneousTrialFESpace(U::SingleFieldFESpace,n::Int)
  dirichlet_values = fill(zero_dirichlet_values(U),n)
  TrialFESpace(dirichlet_values,U)
end

function FESpaces.TrialFESpace!(
  f::TrialFESpace,
  objects::Vector{T}
  ) where {T<:Union{AbstractArray,Function}}

  dir_values = get_dirichlet_dof_values(f)
  dv_cache = first(dir_values)
  dir_values_scratch = zero_dirichlet_values(f)
  for (n,obj) = enumerate(objects)
    dv = copy(dv_cache)
    compute_dirichlet_values_for_tags!(dv,dir_values_scratch,f,obj)
    dir_values[n] = dv
  end
  dir_values
end

function evaluate!(Ut::T,U::ParamTransientTrialFESpace,params::Table,t::Real) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_t = map(o->map(μ->o(μ,t),params),U.dirichlet_μt)
  else
    objects_at_t = map(μ->U.dirichlet_μt(μ,t),params)
  end
  TrialFESpace!(Ut,objects_at_t)
  Ut
end

function Arrays.evaluate(U::ParamTransientTrialFESpace,params::Table,t::Real)
  k = length(params)
  Ut = allocate_trial_space(U,k)
  evaluate!(Ut,U,params,t)
  Ut
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Arrays.evaluate(U::ParamTransientTrialFESpace,::Nothing,::Nothing) = U.Ud0
Arrays.evaluate(U::ParamTransientTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::SingleFieldFESpace)(μ,t) = U
(U::ParamTransientTrialFESpace)(μ,t) = evaluate(U,μ,t)
(U::ParamTransientTrialFESpace)(μ) = evaluate(U,μ)

"""
Time derivative of the Dirichlet functions
"""
∂ₚt(U::ParamTransientTrialFESpace) =
  ParamTransientTrialFESpace(U.space,∂ₚt.(U.dirichlet_μt))

"""
Time 2nd derivative of the Dirichlet functions
"""
∂ₚtt(U::ParamTransientTrialFESpace) =
  ParamTransientTrialFESpace(U.space,∂ₚtt.(U.dirichlet_μt))

# Define the ParamTrialFESpace interface for affine spaces

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

struct ParamTransientMultiFieldTrialFESpace
  spaces::Vector
end
Base.iterate(m::ParamTransientMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::ParamTransientMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::ParamTransientMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamTransientMultiFieldTrialFESpace) = length(m.spaces)

function allocate_trial_space(U::ParamTransientMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμt::T,U::ParamTransientMultiFieldTrialFESpace,μ,t::Real) where T
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Arrays.evaluate(U::ParamTransientMultiFieldTrialFESpace,μ,t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Arrays.evaluate(U::ParamTransientMultiFieldTrialFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([fesp(nothing,nothing) for fesp in U.spaces])
end

(U::ParamTransientMultiFieldTrialFESpace)(μ,t) = evaluate(U,μ,t)
(U::ParamTransientMultiFieldTrialFESpace)(μ) = evaluate(U,μ)

function ∂ₚt(U::ParamTransientMultiFieldTrialFESpace)
  spaces = ∂ₚt.(U.spaces)
  ParamTransientMultiFieldFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector)
  ParamTransientMultiFieldTrialFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end
