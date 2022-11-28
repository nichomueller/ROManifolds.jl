"""
A single field FE space with parametric Dirichlet data (see Multifield below).
"""
struct ParamTrialFESpace{A,B}
  space::A
  dirichlet_μ::Union{Function,Vector{<:Function}}
  Ud0::B

  function ParamTrialFESpace(space::A,dirichlet_μ::Union{Function,Vector{<:Function}}) where A
    Ud0 = HomogeneousTrialFESpace(space)
    B = typeof(Ud0)
    new{A,B}(space,dirichlet_μ,Ud0)
  end
end

function ParamTrialFESpace(space::S) where S
  HomogeneousTrialFESpace(space)
end

"""
Parameter evaluation without allocating Dirichlet vals
"""
function evaluate!(Uμ::T,U::ParamTrialFESpace,μ::Param) where T
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
function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTrialFESpace)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTrialFESpace,μ::Param)
  Uμ = allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Gridap.evaluate(U::ParamTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTrialFESpace)(μ) = Gridap.evaluate(U,μ)

# Define the ParamTrialFESpace interface for affine spaces

function evaluate!(::FESpace,U::FESpace,::Param)
  U
end

function Gridap.evaluate(U::FESpace,::Param)
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
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμ::T,U::ParamMultiFieldTrialFESpace,μ::Param) where T
  spaces_at_μ = [evaluate!(Uμi,Ui,μ) for (Uμi,Ui) in zip(Uμ,U)]
  MultiFieldFESpace(spaces_at_μ)
end

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamMultiFieldTrialFESpace,μ::Param)
  Uμ = allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

function Gridap.evaluate(U::ParamMultiFieldTrialFESpace,::Nothing)
  MultiFieldFESpace([Gridap.evaluate(fesp,nothing) for fesp in U.spaces])
end

(U::ParamMultiFieldTrialFESpace)(μ) = Gridap.evaluate(U,μ)
