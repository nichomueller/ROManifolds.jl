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
function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientTrialFESpace)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function Gridap.evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::AbstractVector,t::Real) where T
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
function Gridap.evaluate(U::ParamTransientTrialFESpace,μ::AbstractVector,t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

"""
Parameter evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTransientTrialFESpace,μ::AbstractVector)
  Uμt = evaluate(U,μ,0.)
  if isa(U.dirichlet_μt,Vector)
    objects_at_t = t -> map(o->o(μ,t),U.dirichlet_μt)
  else
    objects_at_t = t -> U.dirichlet_μt(μ,t)
  end
  TransientTrialFESpace(Uμt,objects_at_t)
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Gridap.evaluate(U::ParamTransientTrialFESpace,::Nothing,::Nothing) = U.Ud0
Gridap.evaluate(U::ParamTransientTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::SingleFieldFESpace)(::AbstractVector,::Real) = U
(U::ParamTransientTrialFESpace)(μ::AbstractVector,t::Real) = Gridap.evaluate(U,μ,t)
(U::ParamTransientTrialFESpace)(μ::AbstractVector) = Gridap.evaluate(U,μ)
(U::ParamTransientTrialFESpace)(::Nothing,::Nothing) = Gridap.evaluate(U,nothing,nothing)
(U::ParamTransientTrialFESpace)(::Nothing) = Gridap.evaluate(U,nothing)

"""
Time derivative of the Dirichlet functions
"""
Gridap.ODEs.TransientFETools.∂t(U::ParamTransientTrialFESpace) =
  ParamTransientTrialFESpace(U.space,∂t.(U.dirichlet_μt))

"""
Time 2nd derivative of the Dirichlet functions
"""
Gridap.ODEs.TransientFETools.∂tt(U::ParamTransientTrialFESpace) =
  ParamTransientTrialFESpace(U.space,∂tt.(U.dirichlet_μt))

# Define the ParamTrialFESpace interface for affine spaces

function Gridap.evaluate!(::FESpace,U::FESpace,::AbstractVector,::Real)
  U
end

function Gridap.evaluate(U::FESpace,::AbstractVector,::Real)
  U
end

function Gridap.evaluate(U::FESpace,::Nothing,::Nothing)
  U
end

# Define the interface for MultiField

struct ParamTransientMultiFieldFESpace
  spaces::Vector
end
Base.iterate(m::ParamTransientMultiFieldFESpace) = iterate(m.spaces)
Base.iterate(m::ParamTransientMultiFieldFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::ParamTransientMultiFieldFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamTransientMultiFieldFESpace) = length(m.spaces)

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientMultiFieldFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate!(Uμt::T,U::ParamTransientMultiFieldFESpace,μ::AbstractVector,t::Real) where T
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientMultiFieldFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamTransientMultiFieldFESpace,μ::AbstractVector,t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Gridap.evaluate(U::ParamTransientMultiFieldFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([Gridap.evaluate(fesp,nothing,nothing) for fesp in U.spaces])
end

(U::TransientMultiFieldTrialFESpace)(::AbstractVector,::Real) = U
(U::ParamTransientMultiFieldFESpace)(μ,t) = Gridap.evaluate(U,μ,t)

function Gridap.ODEs.TransientFETools.∂t(U::ParamTransientMultiFieldFESpace)
  spaces = ∂t.(U.spaces)
  ParamTransientMultiFieldFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector)
  ParamTransientMultiFieldFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function Gridap.FESpaces.get_fe_basis(
  U::ParamTransientMultiFieldFESpace,
  i::Int)

  get_fe_basis(U[i])
end

function Gridap.FESpaces.get_trial_fe_basis(
  U::ParamTransientMultiFieldFESpace,
  i::Int)

  get_trial_fe_basis(U[i])
end
