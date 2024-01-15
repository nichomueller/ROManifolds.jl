"""
A single field FE space with parametric, transient Dirichlet data.
"""
struct TransientTrialPFESpace{S,B}
  space::S
  dirichlet_pt::Union{Function,Vector{<:Function}}
  Ud0::B

  function TransientTrialPFESpace(space::S,dirichlet_pt::Union{Function,Vector{<:Function}}) where S
    Ud0 = allocate_trial_space(space)
    B = typeof(Ud0)
    new{S,B}(space,dirichlet_pt,Ud0)
  end
end

"""
Allocate the space to be used as first argument in evaluate!
"""
function TransientFETools.allocate_trial_space(
  U::TransientTrialPFESpace,
  r::Realization)
  HomogeneousTrialPFESpace(U.space,length(r))
end

function TransientFETools.allocate_trial_space(
  U::TransientTrialPFESpace,
  r::TrivialTransientPRealization)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function Arrays.evaluate!(Upt::T,U::TransientTrialPFESpace,r::Realization) where T
  objects_at_pt = []
  for p in get_parameters(r), t in get_times(r)
    if isa(U.dirichlet_pt,Vector)
      push!(objects_at_pt,map(o->o(p,t),U.dirichlet_pt))
    else
      push!(objects_at_pt,U.dirichlet_pt(p,t))
    end
  end
  TrialPFESpace!(Upt,objects_at_pt)
  Upt
end

function Arrays.evaluate!(
  Upt::T,
  U::TransientTrialPFESpace,
  r::GenericTransientPRealization{<:AbstractVector{<:Number}}) where T
  evaluate!(Upt,U,GenericTransientPRealization([get_parameters(r)],get_times(r)))
end

function Arrays.evaluate!(
  Upt::T,
  U::TransientTrialPFESpace,
  r::TransientPRealization) where T

  p = get_parameters(r)
  t = get_times(r)
  if isa(U.dirichlet_pt,Vector)
    object = map(o->o(p,t),U.dirichlet_pt)
  else
    object = U.dirichlet_pt(p,t)
  end
  TrialFESpace!(Upt,object)
  Upt
end

"""
Parameter, time evaluation allocating Dirichlet vals
"""
function Arrays.evaluate(U::TransientTrialPFESpace,r)
  Upt = allocate_trial_space(U,r)
  evaluate!(Upt,U,r)
  Upt
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Arrays.evaluate(U::TransientTrialPFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::TransientTrialPFESpace)(r) = evaluate(U,r)

"""
Time derivative of the Dirichlet functions
"""
∂ₚt(U::TransientTrialPFESpace) = TransientTrialPFESpace(U.space,∂ₚt.(U.dirichlet_pt))
∂ₚt(U::SingleFieldFESpace) = HomogeneousTrialFESpace(U)
∂ₚt(U::MultiFieldFESpace) = MultiFieldFESpace(∂t.(U.spaces))
∂ₚt(t::T) where T<:Number = zero(T)

"""
Time 2nd derivative of the Dirichlet functions
"""
∂ₚtt(U::TransientTrialPFESpace) = TransientTrialPFESpace(U.space,∂ₚtt.(U.dirichlet_pt))
∂ₚtt(U::SingleFieldFESpace) = HomogeneousTrialFESpace(U)
∂ₚtt(U::MultiFieldFESpace) = MultiFieldFESpace(∂tt.(U.spaces))
∂ₚtt(t::T) where T<:Number = zero(T)

FESpaces.zero_free_values(f::TransientTrialPFESpace) = zero_free_values(f.space)
FESpaces.has_constraints(f::TransientTrialPFESpace) = has_constraints(f.space)
FESpaces.get_dof_value_type(f::TransientTrialPFESpace) = get_dof_value_type(f.space)
FESpaces.get_vector_type(f::TransientTrialPFESpace) = get_vector_type(f.space)

function Arrays.evaluate(U::FESpace,p,t)
  Upt = allocate_trial_space(U,p,t)
  evaluate!(Upt,U,p,t)
end

# Define the TransientTrialFESpace interface for stationary spaces

Arrays.evaluate!(Upt::FESpace,U::FESpace,r::Realization) = U
Arrays.evaluate(U::FESpace,r::Realization) = U

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
  U::TransientMultiFieldTrialPFESpace,
  args...)
  spaces = map(fe->allocate_trial_space(fe,args...),U.spaces)
  MultiFieldPFESpace(spaces)
end

function TransientFETools.allocate_trial_space(
  U::TransientMultiFieldTrialPFESpace,p::Vector,t::Real)
  spaces = map(fe->allocate_trial_space(fe,p,t),U.spaces)
  MultiFieldFESpace(spaces)
end

function Arrays.evaluate!(Upt,U::TransientMultiFieldTrialPFESpace,p,t)
  spaces_at_pt = [evaluate!(Upti,Ui,p,t) for (Upti,Ui) in zip(Upt,U)]
  MultiFieldPFESpace(spaces_at_pt)
end

function Arrays.evaluate!(Upt,U::TransientMultiFieldTrialPFESpace,p::Vector,t::Real)
  spaces_at_pt = [evaluate!(Upti,Ui,p,t) for (Upti,Ui) in zip(Upt,U)]
  MultiFieldFESpace(spaces_at_pt)
end

function Arrays.evaluate(U::TransientMultiFieldTrialPFESpace,p,t)
  Upt = allocate_trial_space(U,p,t)
  evaluate!(Upt,U,p,t)
  Upt
end

function Arrays.evaluate(U::TransientMultiFieldTrialPFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([fesp(nothing) for fesp in U.spaces])
end

(U::TransientMultiFieldTrialPFESpace)(p,t) = evaluate(U,p,t)

function ∂ₚt(U::TransientMultiFieldTrialPFESpace)
  spaces = ∂ₚt.(U.spaces)
  TransientMultiFieldPFESpace(spaces)
end
