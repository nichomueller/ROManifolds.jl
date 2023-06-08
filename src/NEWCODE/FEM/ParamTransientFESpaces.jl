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

function evaluate!(::FESpace,U::FESpace,::AbstractVector,::Real)
  U
end

function Gridap.evaluate(U::FESpace,::AbstractVector,::Real)
  U
end

function Gridap.evaluate(U::FESpace,::Nothing,::Nothing)
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

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμt::T,U::ParamTransientMultiFieldTrialFESpace,μ::AbstractVector,t::Real) where T
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,μ::AbstractVector,t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([Gridap.evaluate(fesp,nothing,nothing) for fesp in U.spaces])
end

(U::TransientMultiFieldTrialFESpace)(::AbstractVector,::Real) = U
(U::ParamTransientMultiFieldTrialFESpace)(μ,t) = Gridap.evaluate(U,μ,t)

function Gridap.ODEs.TransientFETools.∂t(U::ParamTransientMultiFieldTrialFESpace)
  spaces = ∂t.(U.spaces)
  ParamTransientMultiFieldFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector)
  ParamTransientMultiFieldTrialFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

# function Gridap.FESpaces.FEFunction(
#   trial::ParamTransientTrialFESpace,
#   u::AbstractVector,
#   μ::AbstractVector,
#   t::Real)

#   FEFunction(trial(μ,t),u)
# end

# function Gridap.FESpaces.FEFunction(
#   trial::ParamTransientTrialFESpace,
#   u::AbstractMatrix,
#   μ::AbstractVector,
#   times::Vector{<:Real})

#   Nt = length(times)
#   @assert size(u,2) == Nt "Wrong dimensions"

#   n(tθ) = findall(x -> x == tθ,timesθ)[1]
#   tθ -> FEFunction(trial,uk[:,n(tθ)],μ,tθ)
# end
