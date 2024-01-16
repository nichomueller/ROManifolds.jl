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
  r::TrivialRealization)
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
  evaluate!(Upt,U,TransientPRealization([get_parameters(r)],get_times(r)))
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

function Arrays.evaluate(U::FESpace,r::Realization)
  Upt = allocate_trial_space(U,r)
  evaluate!(Upt,U,r)
end

# Define the TransientTrialFESpace interface for stationary spaces

Arrays.evaluate!(Upt::FESpace,U::FESpace,r::Realization) = U
Arrays.evaluate(U::FESpace,r::Realization) = U

# Define the interface for MultiField

struct TransientMultiFieldTrialPFESpace{MS<:MultiFieldStyle,CS<:ConstraintStyle,V}
  vector_type::Type{V}
  spaces::Vector
  multi_field_style::MS
  constraint_style::CS
  function TransientMultiFieldTrialPFESpace(
    ::Type{V},
    spaces::Vector,
    multi_field_style::MultiFieldStyle) where V
    @assert length(spaces) > 0

    MS = typeof(multi_field_style)
    if any( map(has_constraints,spaces) )
      constraint_style = Constrained()
    else
      constraint_style = UnConstrained()
    end
    CS = typeof(constraint_style)
    new{MS,CS,V}(V,spaces,multi_field_style,constraint_style)
  end
end

# Default constructors
function TransientMultiFieldPFESpace(
  spaces::Vector;style = ConsecutiveMultiFieldStyle())
  Ts = map(get_dof_value_type,spaces)
  T  = typeof(*(map(zero,Ts)...))
  if isa(style,BlockMultiFieldStyle)
    style = BlockMultiFieldStyle(style,spaces)
    V = map(spaces) do space
      zero_free_values(space.space)
    end
    VT = typeof(mortar(V))
  else
    VT = Vector{T}
  end
  TransientMultiFieldTrialPFESpace(VT,spaces,style)
end

function TransientMultiFieldPFESpace(::Type{V},spaces::Vector) where V
  TransientMultiFieldTrialPFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

function TransientMultiFieldPFESpace(
  spaces::Vector{<:SingleFieldFESpace};style = ConsecutiveMultiFieldStyle())
  MultiFieldFESpace(spaces,style=style)
end

function TransientMultiFieldPFESpace(
  ::Type{V},spaces::Vector{<:SingleFieldFESpace}) where V
  MultiFieldFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

Base.iterate(m::TransientMultiFieldTrialPFESpace) = iterate(m.spaces)
Base.iterate(m::TransientMultiFieldTrialPFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::TransientMultiFieldTrialPFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::TransientMultiFieldTrialPFESpace) = length(m.spaces)

function Arrays.evaluate!(
  Upt::T,U::TransientMultiFieldTrialPFESpace,r::Realization) where T
  spaces_at_r = [evaluate!(Upti,Ui,r) for (Upti,Ui) in zip(Upt,U)]
  return MultiFieldPFESpace(spaces_at_r;style=MultiFieldStyle(U))
end

function Arrays.evaluate!(
  Upt::T,U::TransientMultiFieldTrialPFESpace,r::TrivialRealization) where T
  spaces_at_r = [evaluate!(Upti,Ui,r) for (Upti,Ui) in zip(Upt,U)]
  return MultiFieldFESpace(spaces_at_r;style=MultiFieldStyle(U))
end

function FESpaces.allocate_trial_space(
  U::TransientMultiFieldTrialPFESpace,r::Realization)
  spaces = map(fe->allocate_trial_space(fe,r),U.spaces)
  return MultiFieldPFESpace(spaces;style=MultiFieldStyle(U))
end

function FESpaces.allocate_trial_space(
  U::TransientMultiFieldTrialPFESpace,r::TrivialRealization)
  spaces = map(fe->allocate_trial_space(fe,r),U.spaces)
  return MultiFieldFESpace(spaces;style=MultiFieldStyle(U))
end

function Arrays.evaluate(U::TransientMultiFieldTrialPFESpace,r::Realization)
  Upt = allocate_trial_space(U)
  evaluate!(Upt,U,r)
  return Upt
end

function Arrays.evaluate(U::TransientMultiFieldTrialPFESpace,::Nothing)
  spaces = [evaluate(fesp,nothing) for fesp in U.spaces]
  MultiFieldFESpace(spaces;style=style=MultiFieldStyle(U))
end

(U::TransientMultiFieldTrialPFESpace)(r) = evaluate(U,r)

function ∂ₚt(U::TransientMultiFieldTrialPFESpace)
  spaces = ∂ₚt.(U.spaces)
  TransientMultiFieldPFESpace(spaces;style=style=MultiFieldStyle(U))
end

function FESpaces.zero_free_values(
  f::TransientMultiFieldTrialPFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_ranges   = get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f.spaces[range])),block_ranges)
  block_vtypes   = map(range->get_vector_type(first(f.spaces[range])),block_ranges)
  array = map(1:length(first(f.spaces).dirichlet_values)) do i
    mortar(map(allocate_vector,block_vtypes,block_num_dofs))
  end
  return array
end

FESpaces.get_dof_value_type(f::TransientMultiFieldTrialPFESpace{MS,CS,V}) where {MS,CS,V} = eltype(V)
FESpaces.get_vector_type(f::TransientMultiFieldTrialPFESpace) = f.vector_type
FESpaces.ConstraintStyle(::Type{TransientMultiFieldTrialPFESpace{S,B,V}}) where {S,B,V} = B()
FESpaces.ConstraintStyle(::TransientMultiFieldTrialPFESpace) = ConstraintStyle(typeof(f))
FESpaces.MultiFieldStyle(::Type{TransientMultiFieldTrialPFESpace{S,B,V}}) where {S,B,V} = S()
FESpaces.MultiFieldStyle(f::TransientMultiFieldTrialPFESpace) = MultiFieldStyle(typeof(f))

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::TransientMultiFieldTrialPFESpace{MS},
  test::TransientMultiFieldTrialPFESpace{MS},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where MS <: BlockMultiFieldStyle

  return BlockSparseMatrixAssembler(
    MultiFieldStyle(test),
    trial,
    test,
    SparseMatrixBuilder(mat),
    ArrayBuilder(vec),
    strategy)
end
