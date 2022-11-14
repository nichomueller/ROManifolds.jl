"""
A single field FE space with parametric Dirichlet data (see Multifield below).
"""
struct ParamTrialFESpace{S}
  space::S
  dirichlet_μ::Union{Function,Vector{<:Function}}

  function ParamTrialFESpace(space::S,dirichlet_μ::Union{Function,Vector{<:Function}}) where S
    new{S}(space,dirichlet_μ)
  end
end

function ParamTrialFESpace(space::S) where S
  HomogeneousTrialFESpace(space)
end

"""
Parameter evaluation without allocating Dirichlet vals
"""
function Gridap.evaluate!(Uμ::T,U::ParamTrialFESpace,μ::PT) where {T,PT}
  if isa(U.dirichlet_μ,Vector)
    objects_at_μ = map(o->o(μ), U.dirichlet_μ)
  else
    objects_at_μ = U.dirichlet_μ(μ)
  end
  TrialFESpace!(Uμ,objects_at_μ)
  Uμ
end

"""
Allocate the space to be used as first argument in evaluate!
"""
function allocate_trial_space(U::ParamTrialFESpace)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTrialFESpace,μ::PT) where PT
  Uμ = allocate_trial_space(U)
  Gridap.evaluate!(Uμ,U,μ)
  Uμ
end

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTrialFESpace)(μ) = Gridap.evaluate(U,μ)

# Define the ParamTrialFESpace interface for affine spaces

function Gridap.evaluate!(::FESpace,U::FESpace,::PT) where PT
  U
end

function Gridap.evaluate(U::FESpace,::PT) where PT
  U
end

# Define the interface for MultiField

struct ParamMultiFieldTrialFESpace
  spaces::Vector
end
Base.iterate(m::ParamMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::ParamMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::ParamMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamMultiFieldTrialFESpace) = length(m.spaces)

function ParamMultiFieldFESpace(spaces::Vector)
  ParamMultiFieldTrialFESpace(spaces)
end

function ParamMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  Gridap.MultiFieldFESpace(spaces)
end

function Gridap.evaluate!(Uμ::T,U::ParamMultiFieldTrialFESpace,μ::PT) where {T,PT}
  spaces_at_μ = [Gridap.evaluate!(Uμi,Ui,μ) for (Uμi,Ui) in zip(Uμ,U)]
  Gridap.MultiFieldFESpace(spaces_at_μ)
end

function allocate_trial_space(U::ParamMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  Gridap.MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamMultiFieldTrialFESpace,μ::PT) where PT
  Uμ = allocate_trial_space(U)
  Gridap.evaluate!(Uμ,U,μ)
  Uμ
end

function Gridap.evaluate(U::ParamMultiFieldTrialFESpace,::Nothing)
  Gridap.MultiFieldFESpace([evaluate(fesp,nothing) for fesp in U.spaces])
end

(U::ParamMultiFieldTrialFESpace)(μ) = Gridap.evaluate(U,μ)



"""
A single field FE space with parametric, transient Dirichlet data.
"""
struct ParamTransientTrialFESpace{S,B}
  space::S
  dirichlet_μt::Union{Function,Vector{<:Function}}
  Ud0_μ::B

  function ParamTransientTrialFESpace(space::S,dirichlet_μt::Union{Function,Vector{<:Function}}) where S
    Ud0_μ = HomogeneousTrialFESpace(space)
    B = typeof(Ud0_μ)
    new{S,B}(space,dirichlet_μt)
  end
end

function ParamTransientTrialFESpace(space::S) where S
  HomogeneousTrialFESpace(space)
end

"""
Parameter, time evaluation without allocating Dirichlet vals
"""
function Gridap.evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::PT,t::Real) where {T,PT}
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = map(o->o(μ,t), U.dirichlet_μt)
  else
    objects_at_μt = U.dirichlet_μ(μ,t)
  end
  TrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

"""
Allocate the space to be used as first argument in evaluate!
"""
function allocate_trial_space(U::ParamTransientTrialFESpace)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTransientTrialFESpace,μ::PT,t::Real) where PT
  Uμt = allocate_trial_space(U)
  Gridap.evaluate!(Uμt,U,μ,t)
  Uμt
end

"""
Parameter evaluation without allocating Dirichlet vals (returns a TransientTrialFESpace)
"""
function Gridap.evaluate!(Uμt::T,U::ParamTrialFESpace,μ::PT) where {T,PT}
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = t -> map(o->o(μ,t), U.dirichlet_μt(μ))
  else
    objects_at_μt = t -> U.dirichlet_μ(μ,t)
  end
  TransientTrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

function TransientTrialFESpace!(space::TransientTrialFESpace,objects)
  dir_values(t) = get_dirichlet_dof_values(space(t))
  dir_values_scratch(t) = zero_dirichlet_values(space(t))
  dir_values(t) = compute_dirichlet_values_for_tags!(dir_values(t),dir_values_scratch(t),space(t),objects(t))
  space
end

"""
Parameter evaluation allocating Dirichlet vals (returns a TransientTrialFESpace)
"""
function Gridap.evaluate(U::ParamTransientTrialFESpace,μ::PT) where PT
  Uμt = allocate_trial_space(U)
  Gridap.evaluate!(Uμt,U,μ)
  Uμt
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
evaluate(U::ParamTransientTrialFESpace,::Nothing) = U.Ud0_μ

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTransientTrialFESpace)(μ,t) = Gridap.evaluate(U,μ,t)
(U::TrialFESpace)(μ,t) = U
(U::ZeroMeanFESpace)(μ,t) = U

"""
Time derivative of the Dirichlet functions
"""
∂t(U::ParamTransientTrialFESpace) = μ -> TransientTrialFESpace(U.space,∂t.(U.dirichlet_μt(μ)))

"""
Time 2nd derivative of the Dirichlet functions
"""
∂tt(U::TransientTrialFESpace) = μ -> TransientTrialFESpace(U.space,∂tt.(U.dirichlet_μt(μ)))

# Define the ParamTrialFESpace interface for affine spaces

function Gridap.evaluate!(::FESpace,U::FESpace,::PT,::Real) where PT
  U
end

function Gridap.evaluate(U::FESpace,::PT,::Real) where PT
  U
end

function Gridap.evaluate(U::FESpace,::PT,::nothing) where PT
  U
end

# Define the interface for MultiField
