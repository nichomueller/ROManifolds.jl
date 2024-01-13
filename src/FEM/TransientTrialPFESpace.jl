"""
A single field FE space with parametric, transient Dirichlet data.
"""
struct TransientTrialPFESpace{S,B}
  space::S
  dirichlet_μt::Union{Function,Vector{<:Function}}
  Ud0::B

  function TransientTrialPFESpace(space::S,dirichlet_μt::Union{Function,Vector{<:Function}}) where S
    Ud0 = allocate_trial_space(space)
    B = typeof(Ud0)
    new{S,B}(space,dirichlet_μt,Ud0)
  end
end

"""
Allocate the space to be used as first argument in evaluate!
"""
function TransientFETools.allocate_trial_space(U::TransientTrialPFESpace,μ,t)
  HomogeneousTrialPFESpace(U.space,_length(μ,t))
end

function TransientFETools.allocate_trial_space(
  U::TransientTrialPFESpace,::Vector{<:Number},::Real)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function Arrays.evaluate!(Ut::T,U::TransientTrialPFESpace,μ,t) where T
  objects_at_μt = []
  for μi in μ, ti in t
    if isa(U.dirichlet_μt,Vector)
      push!(objects_at_μt,map(o->o(μi,ti),U.dirichlet_μt))
    else
      push!(objects_at_μt,U.dirichlet_μt(μi,ti))
    end
  end
  TrialPFESpace!(Ut,objects_at_μt)
  Ut
end

function Arrays.evaluate!(Ut::T,U::TransientTrialPFESpace,μ::Vector{<:Number},t) where T
  objects_at_μt = []
  for ti in t
    if isa(U.dirichlet_μt,Vector)
      push!(objects_at_μt,map(o->o(μ,ti),U.dirichlet_μt))
    else
      push!(objects_at_μt,U.dirichlet_μt(μ,ti))
    end
  end
  TrialPFESpace!(Ut,objects_at_μt)
  Ut
end

function Arrays.evaluate!(Uμt::T,U::TransientTrialPFESpace,μ::Vector{<:Number},t::Real) where T
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
function Arrays.evaluate(U::TransientTrialPFESpace,μ,t)
  Uμt = allocate_trial_space(U,μ,t)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Arrays.evaluate(U::TransientTrialPFESpace,::Nothing,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::SingleFieldFESpace)(μ,t) = evaluate(U,μ,t)
(U::TransientTrialPFESpace)(μ,t) = evaluate(U,μ,t)

"""
Time derivative of the Dirichlet functions
"""
∂ₚt(U::TransientTrialPFESpace) = TransientTrialPFESpace(U.space,∂ₚt.(U.dirichlet_μt))
∂ₚt(U::SingleFieldFESpace) = HomogeneousTrialFESpace(U)
∂ₚt(U::MultiFieldFESpace) = MultiFieldFESpace(∂t.(U.spaces))

"""
Time 2nd derivative of the Dirichlet functions
"""
∂ₚtt(U::TransientTrialPFESpace) = TransientTrialPFESpace(U.space,∂ₚtt.(U.dirichlet_μt))

# Define the TrialPFESpace interface for affine spaces

function TransientFETools.allocate_trial_space(U::FESpace,args...)
  U
end

function TransientFETools.allocate_trial_space(U::FESpace,μ,t)
  HomogeneousTrialFESpace(U)
end

Arrays.evaluate!(Ut::FESpace,::FESpace,μ,t) = Ut
Arrays.evaluate!(::FESpace,U::FESpace,::Vector{<:Number},::Real) = U

function Arrays.evaluate(U::FESpace,μ,t)
  Uμt = allocate_trial_space(U,μ,t)
  evaluate!(Uμt,U,μ,t)
end

Arrays.evaluate(U::FESpace,::Nothing,::Nothing) = U

function FESpaces.SparseMatrixAssembler(
  trial::TransientTrialPFESpace,
  test::FESpace)
  SparseMatrixAssembler(trial(nothing,nothing),test)
end

# Define the interface for MultiField

struct TransientMultiFieldTrialPFESpace
  spaces::Vector
end

Base.iterate(m::TransientMultiFieldTrialPFESpace) = iterate(m.spaces)
Base.iterate(m::TransientMultiFieldTrialPFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::TransientMultiFieldTrialPFESpace,::Colon) = m
Base.getindex(m::TransientMultiFieldTrialPFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::TransientMultiFieldTrialPFESpace) = length(m.spaces)

function TransientMultiFieldPFESpace(spaces::Vector)
  TransientMultiFieldTrialPFESpace(spaces)
end

function TransientMultiFieldPFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function TransientFETools.allocate_trial_space(
  U::TransientMultiFieldTrialPFESpace,args...)
  spaces = map(fe->allocate_trial_space(fe,args...),U.spaces)
  MultiFieldPFESpace(spaces)
end

function TransientFETools.allocate_trial_space(
  U::TransientMultiFieldTrialPFESpace,μ::Vector,t::Real)
  spaces = map(fe->allocate_trial_space(fe,μ,t),U.spaces)
  MultiFieldFESpace(spaces)
end

function Arrays.evaluate!(Uμt,U::TransientMultiFieldTrialPFESpace,μ,t)
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldPFESpace(spaces_at_μt)
end

function Arrays.evaluate!(Uμt,U::TransientMultiFieldTrialPFESpace,μ::Vector,t::Real)
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Arrays.evaluate(U::TransientMultiFieldTrialPFESpace,μ,t)
  Uμt = allocate_trial_space(U,μ,t)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Arrays.evaluate(U::TransientMultiFieldTrialPFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([fesp(nothing,nothing) for fesp in U.spaces])
end

(U::TransientMultiFieldTrialPFESpace)(μ,t) = evaluate(U,μ,t)

function ∂ₚt(U::TransientMultiFieldTrialPFESpace)
  spaces = ∂ₚt.(U.spaces)
  TransientMultiFieldPFESpace(spaces)
end

function FESpaces.SparseMatrixAssembler(
  trial::TransientMultiFieldTrialPFESpace,
  test::FESpace)
  SparseMatrixAssembler(trial(nothing,nothing),test)
end
