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
function Gridap.evaluate(U::ParamTrialFESpace,μ::Vector{Float})
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

function evaluate!(::FESpace,U::FESpace,::Vector{Float})
  U
end

function Gridap.evaluate(U::FESpace,::Vector{Float})
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
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamMultiFieldTrialFESpace,μ::Vector{Float})
  Uμ = allocate_trial_space(U)
  evaluate!(Uμ,U,μ)
  Uμ
end

function Gridap.evaluate(U::ParamMultiFieldTrialFESpace,::Nothing)
  MultiFieldFESpace([Gridap.evaluate(fesp,nothing) for fesp in U.spaces])
end

(U::ParamMultiFieldTrialFESpace)(μ) = Gridap.evaluate(U,μ)

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
Allocate the space to be used as first argument in evaluate!
"""
function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientTrialFESpace)
  HomogeneousTrialFESpace(U.space)
end

"""
Parameter, time evaluation without allocating Dirichlet vals (returns a TrialFESpace)
"""
function evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::Vector{Float},t::Real) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = map(o->o(t,μ), U.dirichlet_μt)
  else
    objects_at_μt = U.dirichlet_μt(t,μ)
  end
  TrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

"""
Parameter evaluation without allocating Dirichlet vals (returns a TransientTrialFESpace)
"""
function evaluate!(Uμt::T,U::ParamTransientTrialFESpace,μ::Vector{Float}) where T
  if isa(U.dirichlet_μt,Vector)
    objects_at_μt = map(o->o(t,μ), U.dirichlet_μt)
  else
    objects_at_μt = U.dirichlet_μt(t,μ)
  end
  TrialFESpace!(Uμt,objects_at_μt)
  Uμt
end

"""
Parameter, time evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTransientTrialFESpace,μ::Vector{Float},t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

"""
Time evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTransientTrialFESpace,t::Real)
  Ut = allocate_trial_space(U)
  evaluate!(Ut,U,t)
  Ut
end

"""
Parameter evaluation allocating Dirichlet vals
"""
function Gridap.evaluate(U::ParamTransientTrialFESpace,μ::Vector{Float})
  Uμ = allocate_trial_space(U)
  evaluate!(Ut,U,μ)
  Uμ
end

"""
We can evaluate at `nothing` when we do not care about the Dirichlet vals
"""
Gridap.evaluate(U::ParamTransientTrialFESpace,::Nothing,::Nothing) = U.Ud0
Gridap.evaluate(U::ParamTransientTrialFESpace,::Nothing) = U.Ud0

"""
Functor-like evaluation. It allocates Dirichlet vals in general.
"""
(U::ParamTransientTrialFESpace)(μ::Vector{Float},t::Real) = Gridap.evaluate(U,μ,t)
(U::ParamTransientTrialFESpace)(μ::Vector{Float}) = Gridap.evaluate(U,μ)
(U::ParamTransientTrialFESpace)(t::Real) = Gridap.evaluate(U,t)

"""
Time derivative of the Dirichlet functions
"""
Gridap.ODEs.TransientFETools.∂t(U::ParamTransientTrialFESpace) =
  ParamTransientTrialFESpace(U.space,∂t.(U.dirichlet_μt))

"""
Time 2nd derivative of the Dirichlet functions
"""
Gridap.ODEs.TransientFETools.∂tt(U::ParamTransientTrialFESpace) =
  ParamTransientTrialFESpace(U.space,∂tt.(U.dirichlet_μt))

# Define the ParamTrialFESpace interface for affine spaces

function evaluate!(::FESpace,U::FESpace,::Vector{Float},::Real)
  U
end

function Gridap.evaluate(U::FESpace,::Vector{Float},::Real)
  U
end

function Gridap.evaluate(U::FESpace,::Vector{Float},::Nothing)
  U
end

# Define the interface for MultiField

struct ParamTransientMultiFieldTrialFESpace
  spaces::Vector
end
Base.iterate(m::ParamTransientMultiFieldTrialFESpace) = iterate(m.spaces)
Base.iterate(m::ParamTransientMultiFieldTrialFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::ParamTransientMultiFieldTrialFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::ParamTransientMultiFieldTrialFESpace) = length(m.spaces)

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function evaluate!(Uμt::T,U::ParamTransientMultiFieldTrialFESpace,μ::Vector{Float},t::Real) where T
  spaces_at_μt = [evaluate!(Uμti,Ui,μ,t) for (Uμti,Ui) in zip(Uμt,U)]
  MultiFieldFESpace(spaces_at_μt)
end

function Gridap.ODEs.TransientFETools.allocate_trial_space(U::ParamTransientMultiFieldTrialFESpace)
  spaces = allocate_trial_space.(U.spaces)
  MultiFieldFESpace(spaces)
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,μ::Vector{Float},t::Real)
  Uμt = allocate_trial_space(U)
  evaluate!(Uμt,U,μ,t)
  Uμt
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,t::Real)
  Uμt = allocate_trial_space(U)
  μ -> evaluate!(Uμt,U,μ,t)
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,μ::Vector{Float})
  Uμt = allocate_trial_space(U)
  t -> evaluate!(Uμt,U,μ,t)
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,::Nothing,::Nothing)
  MultiFieldFESpace([Gridap.evaluate(fesp,nothing) for fesp in U.spaces])
end

function Gridap.evaluate(U::ParamTransientMultiFieldTrialFESpace,::Nothing)
  Gridap.evaluate(U,nothing,nothing)
end

(U::ParamTransientMultiFieldTrialFESpace)(μ::Vector{Float},t::Real) = Gridap.evaluate(U,μ,t)
(U::ParamTransientMultiFieldTrialFESpace)(μ::Vector{Float}) = Gridap.evaluate(U,μ)
(U::ParamTransientMultiFieldTrialFESpace)(t::Real) = Gridap.evaluate(U,t)

function Gridap.ODEs.TransientFETools.∂t(U::ParamTransientMultiFieldTrialFESpace)
  spaces = ∂t.(U.spaces)
  ParamTransientMultiFieldFESpace(spaces)
end

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
  ParamMultiFieldFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamMultiFieldFESpace(spaces::Vector{MyTests})
  ParamMultiFieldFESpace([first(spaces).test,last(spaces).test])
end

function ParamTransientMultiFieldFESpace(spaces::Vector)
  ParamTransientMultiFieldTrialFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamTransientMultiFieldFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTests})
  ParamTransientMultiFieldFESpace([first(spaces).test,last(spaces).test])
end

function Base.zero(
  s::Union{ParamTrialFESpace,TransientTrialFESpace,ParamTransientTrialFESpace,ParamMultiFieldTrialFESpace})
  zero(Gridap.evaluate(s,nothing))
end

#= function Base.zero(
  s::Union{TransientMultiFieldFESpace,ParamTransientMultiFieldFESpace},
  c=nothing)

  r = first(s.spaces).nfree
  isnothing(c) ? zeros(r) : zeros(r,c)
end =#
