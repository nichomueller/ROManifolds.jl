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
function allocate_trial_space(U::PTTrialFESpace,μ,t)
  _length(a) = 1
  _length(a::Table) = length(a)
  NonaffineHomogeneousPTrialFESpace(U.space,_length(μ)*length(t))
end

function allocate_trial_space(U::PTTrialFESpace,::Vector{<:Number},::Real)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function evaluate!(Ut::T,U::PTTrialFESpace,μ,t) where T
  objects_at_μt = []
  for μi in μ, ti in t
    if isa(U.dirichlet_μt,Vector)
      push!(objects_at_μt,map(o->o(μi,ti),U.dirichlet_μt))
    else
      push!(objects_at_μt,U.dirichlet_μt(μi,ti))
    end
  end
  PTrialFESpace!(Ut,objects_at_μt)
  Ut
end

function evaluate!(Ut::T,U::PTTrialFESpace,μ::Vector{<:Number},t) where T
  objects_at_μt = []
  for ti in t
    if isa(U.dirichlet_μt,Vector)
      push!(objects_at_μt,map(o->o(μ,ti),U.dirichlet_μt))
    else
      push!(objects_at_μt,U.dirichlet_μt(μ,ti))
    end
  end
  PTrialFESpace!(Ut,objects_at_μt)
  Ut
end

function evaluate!(Uμt::T,U::PTTrialFESpace,μ::Vector{<:Number},t::Real) where T
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
function Arrays.evaluate(U::PTTrialFESpace,μ,t)
  Uμt = allocate_trial_space(U,μ,t)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Arrays.evaluate(U::PTTrialFESpace,::Nothing,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::SingleFieldFESpace)(μ,t) = evaluate(U,μ,t)
(U::PTTrialFESpace)(μ,t) = evaluate(U,μ,t)

"""
Time derivative of the Dirichlet functions
"""
∂ₚt(U::PTTrialFESpace) = PTTrialFESpace(U.space,∂ₚt.(U.dirichlet_μt))
∂ₚt(U::SingleFieldFESpace) = HomogeneousTrialFESpace(U)
∂ₚt(U::MultiFieldFESpace) = MultiFieldFESpace(∂t.(U.spaces))
∂ₚt(f::Union{Gridap.ODEs.TransientFETools.TransientCellField,Gridap.ODEs.TransientFETools.TransientFEBasis}) = ∂t(f)
∂ₚt(f::Gridap.ODEs.TransientFETools.TransientMultiFieldCellField) = ∂t(f)

"""
Time 2nd derivative of the Dirichlet functions
"""
∂ₚtt(U::PTTrialFESpace) =
  PTTrialFESpace(U.space,∂ₚtt.(U.dirichlet_μt))

# Define the PTrialFESpace interface for affine spaces

function allocate_trial_space(U::FESpace,args...)
  U
end

function allocate_trial_space(U::FESpace,μ,t)
  _length(a) = 1
  _length(a::Table) = length(a)
  if isa(μ,Vector{<:Number}) && isa(t,Real)
    U
  else
    AffineHomogeneousPTrialFESpace(U,_length(μ)*length(t))
  end
end

Arrays.evaluate!(Ut::FESpace,::FESpace,μ,t) = Ut
Arrays.evaluate!(::FESpace,U::FESpace,::Vector{<:Number},::Real) = U

function Arrays.evaluate(U::FESpace,μ,t)
  Uμt = allocate_trial_space(U,μ,t)
  evaluate!(Uμt,U,μ,t)
end

Arrays.evaluate(U::FESpace,::Nothing,::Nothing) = U

# Define the interface for MultiField

struct PTMultiFieldTrialFESpace
  spaces::Vector
end

Base.iterate(m::PTMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::PTMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::PTMultiFieldTrialFESpace,::Colon) = m
Base.getindex(m::PTMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::PTMultiFieldTrialFESpace) = length(m.spaces)

function PTMultiFieldFESpace(spaces::Vector)
  PTMultiFieldTrialFESpace(spaces)
end

function PTMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function allocate_trial_space(U::PTMultiFieldTrialFESpace,args...)
  spaces = map(fe->allocate_trial_space(fe,args...),U.spaces)
  PMultiFieldFESpace(spaces)
end

function allocate_trial_space(U::PTMultiFieldTrialFESpace,μ::Vector,t::Real)
  spaces = map(fe->allocate_trial_space(fe,μ,t),U.spaces)
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμt,U::PTMultiFieldTrialFESpace,μ,t)
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  PMultiFieldFESpace(spaces_at_μt)
end

function evaluate!(Uμt,U::PTMultiFieldTrialFESpace,μ::Vector,t::Real)
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Arrays.evaluate(U::PTMultiFieldTrialFESpace,μ,t)
  Uμt = allocate_trial_space(U,μ,t)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Arrays.evaluate(U::PTMultiFieldTrialFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([fesp(nothing,nothing) for fesp in U.spaces])
end

(U::PTMultiFieldTrialFESpace)(μ,t) = evaluate(U,μ,t)

function ∂ₚt(U::PTMultiFieldTrialFESpace)
  spaces = ∂ₚt.(U.spaces)
  PTMultiFieldFESpace(spaces)
end
