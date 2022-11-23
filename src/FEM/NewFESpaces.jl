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
function evaluate!(Uμ::T,U::ParamTrialFESpace,μ::Vector{Float}) where T
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
function Gridap.ODEs.TransientFETools.evaluate(U::ParamTrialFESpace,μ::Vector{Float})
  Uμ = Gridap.ODEs.TransientFETools.allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Gridap.ODEs.TransientFETools.evaluate(U::ParamTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTrialFESpace)(μ) = Gridap.ODEs.TransientFETools.evaluate(U,μ)

# Define the ParamTrialFESpace interface for affine spaces

function evaluate!(::FESpace,U::FESpace,::Vector{Float})
  U
end

function Gridap.ODEs.TransientFETools.evaluate(U::FESpace,::Vector{Float})
  U
end

Gridap.FESpaces.get_free_dof_ids(f::ParamTrialFESpace) = get_dirichlet_dof_ids(f.space)
Gridap.FESpaces.num_free_dofs(f::ParamTrialFESpace) = length(get_free_dof_ids(f))

# Define the interface for MultiField

struct ParamMultiFieldTrialFESpace
  spaces::Vector
end
Base.iterate(m::ParamMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::ParamMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::ParamMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamMultiFieldTrialFESpace) = length(m.spaces)
function Gridap.FESpaces.num_free_dofs(m::ParamMultiFieldTrialFESpace)
  n = 0
  for U in m.spaces
    n += num_free_dofs(U)
  end
  n
end

function evaluate!(Uμ::T,U::ParamMultiFieldTrialFESpace,μ::Vector{Float}) where T
  spaces_at_μ = [evaluate!(Uμi,Ui,μ) for (Uμi,Ui) in zip(Uμ,U)]
  MultiFieldFESpace(spaces_at_μ)
end

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamMultiFieldTrialFESpace)
  spaces = Gridap.ODEs.TransientFETools.allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.ODEs.TransientFETools.evaluate(U::ParamMultiFieldTrialFESpace,μ::Vector{Float})
  Uμ = Gridap.ODEs.TransientFETools.allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

function Gridap.ODEs.TransientFETools.evaluate(U::ParamMultiFieldTrialFESpace,::Nothing)
  MultiFieldFESpace([Gridap.ODEs.TransientFETools.evaluate(fesp,nothing) for fesp in U.spaces])
end

(U::ParamMultiFieldTrialFESpace)(μ) = Gridap.ODEs.TransientFETools.evaluate(U,μ)

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
Parameter, time evaluation without allocating Dirichlet vals
"""
function evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::Vector{Float},t::Real) where T
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
function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientTrialFESpace)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation allocating Dirichlet vals
"""
function Gridap.ODEs.TransientFETools.evaluate(U::ParamTransientTrialFESpace,μ::Vector{Float},t::Real)
  Uμt = Gridap.ODEs.TransientFETools.allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

"""
Parameter evaluation without allocating Dirichlet vals (returns a TransientTrialFESpace)
"""
function evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::Vector{Float}) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = t -> map(o->o(μ,t), U.dirichlet_μt(μ))
  else
    objects_at_μt = t -> U.dirichlet_μ(μ,t)
  end
  TransientTrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

function TransientTrialFESpace!(space::TransientTrialFESpace,objects)
  dir_values_scratch(t) = zero_dirichlet_values(space(t))
  dir_values(t) = compute_dirichlet_values_for_tags!(
    get_dirichlet_dof_values(space(t)),dir_values_scratch(t),space(t),objects(t))
  space
end

"""
Parameter evaluation allocating Dirichlet vals (returns a TransientTrialFESpace)
"""
function Gridap.ODEs.TransientFETools.evaluate(U::ParamTransientTrialFESpace,μ::Vector{Float})
  Uμt = Gridap.ODEs.TransientFETools.allocate_trial_space(U)
  evaluate!(Uμt,U,μ)
  Uμt
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Gridap.ODEs.TransientFETools.evaluate(U::ParamTransientTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTransientTrialFESpace)(μ,t) = Gridap.ODEs.TransientFETools.evaluate(U,μ,t)
(U::TrialFESpace)(μ,t) = U
(U::ZeroMeanFESpace)(μ,t) = U

"""
Time derivative of the Dirichlet functions
"""
∂t(U::ParamTransientTrialFESpace) = μ -> TransientTrialFESpace(U.space,∂t.(U.dirichlet_μt(μ)))

"""
Time 2nd derivative of the Dirichlet functions
"""
∂tt(U::ParamTransientTrialFESpace) = μ -> TransientTrialFESpace(U.space,∂tt.(U.dirichlet_μt(μ)))

# Define the ParamTrialFESpace interface for affine spaces

function evaluate!(::FESpace,U::FESpace,::Vector{Float},::Real)
  U
end

function Gridap.ODEs.TransientFETools.evaluate(U::FESpace,::Vector{Float},::Real)
  U
end

function Gridap.ODEs.TransientFETools.evaluate(U::FESpace,::Vector{Float},::Nothing)
  U
end

# Define the interface for MultiField




abstract type MySpaces end

struct MyTests <: MySpaces
  test::UnconstrainedFESpace
  test_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTests(
    test::UnconstrainedFESpace,
    test_no_bc::UnconstrainedFESpace)

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(test,test_no_bc)
    new(test,test_no_bc,ddofs_on_full_trian)
  end
end

function MyTests(model,reffe;kwargs...)
  test = TestFESpace(model,reffe;kwargs...)
  test_no_bc = FESpace(model,reffe)
  MyTests(test,test_no_bc)
end

function MyTrial(test::UnconstrainedFESpace)
  HomogeneousTrialFESpace(test)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{true})
  ParamTrialFESpace(test,Gμ.f)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{false})
  ParamTransientTrialFESpace(test,Gμ.f)
end

struct MyTrials{TT} <: MySpaces
  trial::TT
  trial_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTrials(
    trial::TT,
    trial_no_bc::UnconstrainedFESpace) where TT

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(trial.space,trial_no_bc)
    new{TT}(trial,trial_no_bc,ddofs_on_full_trian)
  end
end

function MyTrials(tests::MyTests,args...)
  trial = MyTrial(tests.test,args...)
  trial_no_bc = TrialFESpace(tests.test_no_bc)
  MyTrials(trial,trial_no_bc)
end

function ParamMultiFieldFESpace(spaces::Vector)
  ParamMultiFieldTrialFESpace(spaces)
end

function ParamMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function ParamMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamMultiFieldTrialFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamMultiFieldFESpace(spaces::Vector{MyTests})
  MultiFieldFESpace([first(spaces).test,last(spaces).test])
end

function Base.zero(
  s::Union{ParamTrialFESpace,TransientTrialFESpace,ParamTransientTrialFESpace,ParamMultiFieldTrialFESpace})
  zero(Gridap.ODEs.TransientFETools.evaluate(s,nothing))
end

#= function Base.zero(
  s::Union{TransientMultiFieldFESpace,ParamTransientMultiFieldFESpace},
  c=nothing)

  r = first(s.spaces).nfree
  isnothing(c) ? zeros(r) : zeros(r,c)
end =#

#= function Gridap.MultiField.restrict_to_field(
  f::ParamMultiFieldTrialFESpace,
  free_values::AbstractVector,
  field::Integer)

  offsets = Gridap.MultiField.compute_field_offsets(f)
  U = f.spaces
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  SubVector(free_values,pini,pend)
end

function Gridap.MultiField.compute_field_offsets(f::ParamMultiFieldTrialFESpace)
  U = f.spaces
  n = length(U)
  offsets = zeros(Int,n)
  for i in 1:(n-1)
    Ui = U[i]
    offsets[i+1] = offsets[i] + num_free_dofs(Ui)
  end
  offsets
end

function Gridap.FESpaces.FEFunction(fe::ParamTrialFESpace,free_values)
  diri_values = get_dirichlet_dof_values(fe)
  FEFunction(fe,free_values,diri_values)
end

function Gridap.FESpaces.FEFunction(fe::ParamMultiFieldTrialFESpace,free_values)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = Gridap.MultiField.restrict_to_field(fe,free_values,i)
    FEFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end

function Gridap.FESpaces.EvaluationFunction(fe::ParamTrialFESpace,free_values)
  diri_values = get_dirichlet_dof_values(fe)
  FEFunction(fe,free_values,diri_values)
end

function Gridap.FESpaces.EvaluationFunction(fe::ParamMultiFieldTrialFESpace,free_values)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = Gridap.MultiField.restrict_to_field(fe,free_values,i)
    EvaluationFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end =#
