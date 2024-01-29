"""
A single field FE space with parametric, transient Dirichlet data.
"""
struct TransientTrialParamFESpace{S,B}
  space::S
  dirichlet_pt::Union{Function,Vector{<:Function}}
  Ud0::B

  function TransientTrialParamFESpace(
    space::S,
    dirichlet_pt::Union{Function,Vector{<:Function}}) where S

    Ud0 = allocate_trial_space(space)
    B = typeof(Ud0)
    new{S,B}(space,dirichlet_pt,Ud0)
  end
end

"""
Allocate the space to be used as first argument in evaluate!
"""

function TransientFETools.allocate_trial_space(
  U::TransientTrialParamFESpace,params,times)
  HomogeneousTrialParamFESpace(U.space,Val(length(params)*length(times)))
end

function TransientFETools.allocate_trial_space(
  U::TransientTrialParamFESpace,r::TransientParamRealization)
  allocate_trial_space(U,get_params(r),get_times(r))
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function Arrays.evaluate!(
  Upt::T,
  U::TransientTrialParamFESpace,
  params,
  times) where T

  if isa(U.dirichlet_pt,Vector)
    objects_at_pt = map(o->o(params,times),U.dirichlet_pt)
  else
    objects_at_pt = U.dirichlet_pt(params,times)
  end
  TrialParamFESpace!(Upt,objects_at_pt)
  Upt
end

function Arrays.evaluate!(Upt::T,U::TransientTrialParamFESpace,r::TransientParamRealization) where T
  evaluate!(Upt,U,get_params(r),get_times(r))
end

"""
Parameter, time evaluation allocating Dirichlet vals
"""
function Arrays.evaluate(U::TransientTrialParamFESpace,args...)
  Upt = allocate_trial_space(U,args...)
  evaluate!(Upt,U,args...)
  Upt
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Arrays.evaluate(U::TransientTrialParamFESpace,params::Nothing,times::Nothing) = U.Ud0
Arrays.evaluate(U::TransientTrialParamFESpace,r::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::TransientTrialParamFESpace)(params,times) = evaluate(U,params,times)
(U::TransientTrialParamFESpace)(r) = evaluate(U,r)
(U::TrialFESpace)(params,times) = U
(U::ZeroMeanFESpace)(params,times) = U
"""
Time derivative of the Dirichlet functions
"""
function ODETools.∂t(U::TransientTrialParamFESpace)
  ∂tdir(μ,t) = ∂t.(U.dirichlet_pt(μ,t))
  TransientTrialParamFESpace(U.space,∂tdir)
end

"""
Time 2nd derivative of the Dirichlet functions
"""
function ODETools.∂tt(U::TransientTrialParamFESpace)
  ∂ttdir(μ,t) = ∂tt.(U.dirichlet_pt(μ,t))
  TransientTrialParamFESpace(U.space,∂ttdir)
end

FESpaces.zero_free_values(f::TransientTrialParamFESpace) = @notimplemented
FESpaces.has_constraints(f::TransientTrialParamFESpace) = has_constraints(f.space)
FESpaces.get_dof_value_type(f::TransientTrialParamFESpace) = get_dof_value_type(f.space)
FESpaces.get_vector_type(f::TransientTrialParamFESpace) = @notimplemented

# Define the TransientTrialFESpace interface for stationary spaces

Arrays.evaluate!(Upt::FESpace,U::FESpace,params,times) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,r::TransientParamRealization) = U
Arrays.evaluate(U::FESpace,params,times) = U
Arrays.evaluate(U::FESpace,r::TransientParamRealization) = U
TransientFETools.allocate_trial_space(U::FESpace,args...) = U

# Define the interface for MultiField

struct TransientMultiFieldTrialParamFESpace{MS<:MultiFieldStyle,CS<:ConstraintStyle,V}
  vector_type::Type{V}
  spaces::Vector
  multi_field_style::MS
  constraint_style::CS
  function TransientMultiFieldTrialParamFESpace(
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
function TransientMultiFieldParamFESpace(
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
  TransientMultiFieldTrialParamFESpace(VT,spaces,style)
end

function TransientMultiFieldParamFESpace(::Type{V},spaces::Vector) where V
  TransientMultiFieldTrialParamFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

function TransientMultiFieldParamFESpace(
  spaces::Vector{<:SingleFieldFESpace};style = ConsecutiveMultiFieldStyle())
  MultiFieldFESpace(spaces,style=style)
end

function TransientMultiFieldParamFESpace(
  ::Type{V},spaces::Vector{<:SingleFieldFESpace}) where V
  MultiFieldFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

Base.iterate(m::TransientMultiFieldTrialParamFESpace) = iterate(m.spaces)
Base.iterate(m::TransientMultiFieldTrialParamFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::TransientMultiFieldTrialParamFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::TransientMultiFieldTrialParamFESpace) = length(m.spaces)

function TransientFETools.allocate_trial_space(
  U::TransientMultiFieldTrialParamFESpace,args...)
  spaces = map(fe->allocate_trial_space(fe,args...),U.spaces)
  return MultiFieldParamFESpace(spaces;style=MultiFieldStyle(U))
end

function Arrays.evaluate!(
  Upt::T,U::TransientMultiFieldTrialParamFESpace,args...) where T
  spaces_at_r = [evaluate!(Upti,Ui,args...) for (Upti,Ui) in zip(Upt,U)]
  return MultiFieldParamFESpace(spaces_at_r;style=MultiFieldStyle(U))
end

function Arrays.evaluate(U::TransientMultiFieldTrialParamFESpace,args...)
  Upt = allocate_trial_space(U,args...)
  evaluate!(Upt,U,args...)
  return Upt
end

function Arrays.evaluate(U::TransientMultiFieldTrialParamFESpace,::Nothing,::Nothing)
  spaces = [evaluate(fesp,nothing,nothing) for fesp in U.spaces]
  MultiFieldFESpace(spaces;style=style=MultiFieldStyle(U))
end

function Arrays.evaluate(U::TransientMultiFieldTrialParamFESpace,::Nothing)
  spaces = [evaluate(fesp,nothing) for fesp in U.spaces]
  MultiFieldFESpace(spaces;style=style=MultiFieldStyle(U))
end

(U::TransientMultiFieldTrialParamFESpace)(p,t) = evaluate(U,p,t)
(U::TransientMultiFieldTrialParamFESpace)(r) = evaluate(U,r)

function ODETools.∂t(U::TransientMultiFieldTrialParamFESpace)
  spaces = ∂t.(U.spaces)
  TransientMultiFieldParamFESpace(spaces;style=style=MultiFieldStyle(U))
end

ODETools.∂tt(U::TransientMultiFieldTrialParamFESpace) = ∂t(∂t(U))

function FESpaces.zero_free_values(
  f::TransientMultiFieldTrialParamFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  @notimplemented
end

FESpaces.get_dof_value_type(f::TransientMultiFieldTrialParamFESpace{MS,CS,V}) where {MS,CS,V} = eltype(V)
FESpaces.get_vector_type(f::TransientMultiFieldTrialParamFESpace) = @notimplemented
FESpaces.ConstraintStyle(::Type{TransientMultiFieldTrialParamFESpace{S,B,V}}) where {S,B,V} = B()
FESpaces.ConstraintStyle(::TransientMultiFieldTrialParamFESpace) = ConstraintStyle(typeof(f))
MultiField.MultiFieldStyle(::Type{TransientMultiFieldTrialParamFESpace{S,B,V}}) where {S,B,V} = S()
MultiField.MultiFieldStyle(f::TransientMultiFieldTrialParamFESpace) = MultiFieldStyle(typeof(f))

function TransientFETools.test_transient_trial_fe_space(Uh,μ)
  UhX = evaluate(Uh,nothing)
  @test isa(UhX,FESpace)
  Uh0 = allocate_trial_space(Uh,μ,0.0)
  Uh0 = evaluate!(Uh0,Uh,μ,0.0)
  @test isa(Uh0,FESpace)
  Uh0 = evaluate(Uh,μ,0.0)
  @test isa(Uh0,FESpace)
  Uh0 = Uh(μ,0.0)
  @test isa(Uh0,FESpace)
  Uht=∂t(Uh)
  Uht0=Uht(μ,0.0)
  @test isa(Uht0,FESpace)
  true
end
