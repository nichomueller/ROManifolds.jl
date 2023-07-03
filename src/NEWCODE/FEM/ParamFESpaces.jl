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
function Gridap.evaluate!(Uμ::T,U::ParamTrialFESpace,μ::AbstractVector) where T
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
function Gridap.evaluate(U::ParamTrialFESpace,μ::AbstractVector)
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

function Gridap.evaluate!(::FESpace,U::FESpace,::AbstractVector)
  U
end

function Gridap.evaluate(U::FESpace,::AbstractVector)
  U
end

# Define the interface for MultiField

struct ParamMultiFieldFESpace
  spaces::Vector
end
Base.iterate(m::ParamMultiFieldFESpace) = iterate(m.spaces)
Base.iterate(m::ParamMultiFieldFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::ParamMultiFieldFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamMultiFieldFESpace) = length(m.spaces)

function ParamMultiFieldFESpace(spaces::Vector)
  ParamMultiFieldFESpace(spaces)
end

function ParamMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate!(Uμ::T,U::ParamMultiFieldFESpace,μ::AbstractVector) where T
  spaces_at_μ = [evaluate!(Uμi,Ui,μ) for (Uμi,Ui) in zip(Uμ,U)]
  MultiFieldFESpace(spaces_at_μ)
end

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamMultiFieldFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamMultiFieldFESpace,μ::AbstractVector)
  Uμ = allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

function Gridap.evaluate(U::ParamMultiFieldFESpace,::Nothing)
  MultiFieldFESpace([Gridap.evaluate(fesp,nothing) for fesp in U.spaces])
end

(U::MultiFieldFESpace)(::AbstractVector) = U
(U::ParamMultiFieldFESpace)(μ) = Gridap.evaluate(U,μ)


Gridap.FESpaces.get_fe_basis(U,args...) = get_fe_basis(U)
Gridap.FESpaces.get_trial_fe_basis(U,args...) = get_trial_fe_basis(U)

for Tsp in (:MultiFieldFESpace,:ParamMultiFieldFESpace)
  @eval begin
    Gridap.FESpaces.get_fe_basis(U::$Tsp,i::Int) = get_fe_basis(U[i])
    Gridap.FESpaces.get_trial_fe_basis(U::$Tsp,i::Int) = get_trial_fe_basis(U[i])
  end
end
