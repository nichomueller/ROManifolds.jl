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
function evaluate!(Uμ::T,U::ParamTrialFESpace,μ::AbstractVector) where T
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
function evaluate(U::ParamTrialFESpace,μ::AbstractVector)
  Uμ = allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
evaluate(U::ParamTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTrialFESpace)(μ) = evaluate(U,μ)

# Define the ParamTrialFESpace interface for affine spaces

function Gridap.Arrays.evaluate!(::FESpace,U::FESpace,::AbstractVector)
  U
end

function evaluate(U::FESpace,::AbstractVector)
  U
end

# Define the interface for MultiField

struct ParamMultiFieldTrialFESpace
  spaces::Vector
end
Base.iterate(m::ParamMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::ParamMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(U,args...) = U
Base.getindex(m::ParamMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamMultiFieldTrialFESpace) = length(m.spaces)

function ParamMultiFieldFESpace(spaces::Vector)
  ParamMultiFieldTrialFESpace(spaces)
end

function ParamMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμ::T,U::ParamMultiFieldTrialFESpace,μ::AbstractVector) where T
  spaces_at_μ = [evaluate!(Uμi,Ui,μ) for (Uμi,Ui) in zip(Uμ,U)]
  MultiFieldFESpace(spaces_at_μ)
end

function allocate_trial_space(U::ParamMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function evaluate(U::ParamMultiFieldTrialFESpace,μ::AbstractVector)
  Uμ = allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

function evaluate(U::ParamMultiFieldTrialFESpace,::Nothing)
  MultiFieldFESpace([fesp(nothing) for fesp in U.spaces])
end

(U::MultiFieldFESpace)(::AbstractVector) = U
(U::ParamMultiFieldTrialFESpace)(μ) = evaluate(U,μ)

function _split_solutions(::TrialFESpace,u::AbstractVector)
  u
end

function _split_solutions(trial::MultiFieldFESpace,u::AbstractVector)
  map(1:length(trial.spaces)) do i
    restrict_to_field(trial,u,i)
  end
end
