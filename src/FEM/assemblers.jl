function assemble_mass(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_mass(t)
    if !FEMInfo.probl_nl["M"]
      assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.mₛ*FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.m(t)*FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  return unsteady_mass

end

function assemble_mass(
  ::NTuple{2,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  assemble_mass(get_NTuple(1, Int), FEMSpace, FEMInfo, Param)

end

function assemble_mass(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_mass(t)
    if !FEMInfo.probl_nl["M"]
      assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Param.mₛ*FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Param.m(t)*FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  return unsteady_mass

end

function assemble_mass(
  ::NTuple{4,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  assemble_mass(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["A"]
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_stiffness(t)
    if !FEMInfo.probl_nl["A"]
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.αₛ*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  unsteady_stiffness

end

function assemble_stiffness(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRSteady,
  ::SteadyInfo,
  Param::ParametricInfoSteady)

  ####### USING PECHLET STABILIZATION OF α #######
  α_stab = get_α_stab(FEMSpace, Param)
  ################################################

  assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(α_stab*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)

end

function assemble_stiffness(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRUnsteady,
  ::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  ####### USING PECHLET STABILIZATION OF α #######
  α_stab = get_α_stab(FEMSpace, Param)
  ################################################

  function unsteady_stiffness(t)
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(α_stab(t)*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
  end

  return unsteady_stiffness

end

function assemble_stiffness(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesSteady,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["A"]
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_stiffness(t)
    if !FEMInfo.probl_nl["A"]
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.αₛ*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  unsteady_stiffness

end

function assemble_stiffness(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_stiffness(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_primal_op(
  ::NTuple{3,Int},
  FEMSpace::SteadyProblem)

  assemble_matrix(∫(FEMSpace.ψᵧ*∇⋅(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.Q₀)

end

function assemble_primal_op(
  ::NTuple{4,Int},
  FEMSpace::UnsteadyProblem)

  function unsteady_primal_form(t)
    assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
    FEMSpace.V(t), FEMSpace.Q₀)
  end

  unsteady_primal_form

end

function assemble_primal_op(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem)

  assemble_primal_op(get_NTuple(3, Int), FEMSpace)

end

function assemble_advection(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRSteady,
  ::SteadyInfo,
  Param::ParametricInfoSteady)

  assemble_matrix(∫(FEMSpace.ϕᵥ * (Param.b⋅∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.V₀)

end

function assemble_advection(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function advection(t)
    if !FEMInfo.probl_nl["b"]
      assemble_matrix(∫(FEMSpace.ϕᵥ * (Param.bₛ⋅∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(FEMSpace.ϕᵥ * (Param.b(t)⋅∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
        FEMSpace.V(t), FEMSpace.V₀)
    end
  end

end

function assemble_convection(
  ::NTuple{4,Int},
  FEMSpace::FEMSpaceNavierStokesSteady,
  Param::ParametricInfoSteady)

  C(u) = Param.Re * assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    ((∇FEMSpace.ϕᵤ')⋅u) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  C

end

function assemble_convection(
  ::NTuple{4,Int},
  FEMSpace::FEMSpaceNavierStokesUnsteady,
  Param::ParametricInfoUnsteady)

  C(u,t) = Param.Re * assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    ((∇FEMSpace.ϕᵤ')⋅u) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  C

end

function assemble_reaction(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRSteady,
  ::SteadyInfo,
  Param::ParametricInfoSteady)

  assemble_matrix(∫(FEMSpace.ϕᵥ * FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)

end

function assemble_reaction(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function advection(t)
    if !FEMInfo.probl_nl["R"]
      assemble_matrix(∫(Param.σₛ * FEMSpace.ϕᵥ * FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(Param.σ(t) * FEMSpace.ϕᵥ * FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
        FEMSpace.V(t), FEMSpace.V₀)
    end
  end

end

#= function assemble_SUPG_term(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRSteady,
  Param::ParametricInfoSteady)

  #SUPG STABILIZATION, SET ρ = 1 IF GLS STABILIZATION
  ρ = 0
  b₂(x) = 0.5*Param.b(x)
  div_b₂ = ∇⋅(b₂)
  div_b₂_σ(x) = div_b₂(x) + Param.σ(x)

  factor₁(u) = - Param.α*(Δ(u)) + ∇⋅(Param.b*u) + Param.σ*u - Param.f

  lₛ(v) = - Param.α*(Δ(v)) + div_b₂_σ*v
  lₛₛ(v) = Param.b⋅∇(v) + div_b₂*v
  h = get_h(FEMSpace)
  Pechlet(x) = norm(Param.b(x))*h / (2*Param.α(x))
  ξ(x) = coth(x) - 1/x
  τ(x) = h*ξ(pechlet(x)) / (2*norm(Param.b(x)))

  factor₂(v) = τ * (lₛ(v) + ρ*lₛₛ(v))

  l_stab(u,v) = ∫(factor₁(u)*factor₂(v)) * FEMSpace.dΩ
  L = assemble_matrix(l_stab(FEMSpace.ϕᵤ,FEMSpace.ϕᵥ),
    FEMSpace.V, FEMSpace.V₀)

  l_stab, L

end

function assemble_SUPG_term(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceADRUnsteady,
  Param::ParametricInfoUnsteady)

  #SUPG STABILIZATION, SET ρ = 1 IF GLS STABILIZATION
  ρ = 0
  b₂(x,t::Real) = 0.5*Param.b(x,t)
  b₂(t::Real) = x -> b₂(x,t)
  div_b₂(t::Real) = ∇⋅(b₂(t))
  div_b₂_σ(x,t::Real) = div_b₂(t)(x) + Param.σ(x,t)
  div_b₂_σ(t::Real) = x -> div_b₂_σ(x,t)

  factor₁(t,u) = - Param.α(t)*(Δ(u)) + ∇⋅(Param.b(t)*u) + Param.σ(t)*u - Param.f(t)

  lₛ(t,v) = - Param.α(t)*(Δ(v)) + div_b₂_σ(t)*v
  lₛₛ(t,v) = Param.b(t)⋅∇(v) + div_b₂(t)*v
  h = get_h(FEMSpace)
  Pechlet(x,t::Real) = norm(Param.b(x,t))*h / (2*Param.α(x,t))
  ξ(x) = coth(x) - 1/x
  τ(x,t::Real) = h*ξ(pechlet(x,t)) / (2*norm(Param.b(x,t)))
  τ(t::Real) = x -> τ(x,t)
  factor₂(t,v) = τ(t) * (lₛ(t,v) + ρ*lₛₛ(t,v))

  l_stab(t,u,v) = ∫(factor₁(t,u)*factor₂(t,v)) * FEMSpace.dΩ
  L(t) = assemble_matrix(l_stab(t,FEMSpace.ϕᵤ(t),FEMSpace.ϕᵥ),
    FEMSpace.V(t), FEMSpace.V₀)

  l_stab, L

end =#

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["f"] && !FEMInfo.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::FEMSpacePoissonUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_forcing(t)
    if !FEMInfo.probl_nl["f"]
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
    else
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.f(t))*FEMSpace.dΩ, FEMSpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_forcing(
  ::NTuple{2,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_forcing(get_NTuple(1, Int), FEMSpace, FEMInfo, Param)

end

function assemble_forcing(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokessteady{D},
  FEMInfo::steadyInfo,
  Param::ParametricInfosteady) where D

  if !FEMInfo.probl_nl["f"]
    fₛ(x) = x -> one(VectorValue(FEMInfo.D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_forcing(t)
    if !FEMInfo.probl_nl["f"]
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
    else
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f(t))*FEMSpace.dΩ, FEMSpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_forcing(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_forcing(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_dirichlet_datum(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)

  if nonlin_lift ≤ 1
    return interpolate_dirichlet(x->1., FEMSpace.V)
  else
    return interpolate_dirichlet(Param.g(t), FEMSpace.V)
  end

end

function assemble_dirichlet_datum(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)

  function dirichlet_datum(t)
    if nonlin_lift ≤ 1
      return interpolate_dirichlet(Param.gₛ(t), FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
    end
  end

  dirichlet_datum

end

function assemble_dirichlet_datum(
  ::NTuple{2,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_dirichlet_datum(get_NTuple(1, Int), FEMSpace, FEMInfo, Param)

end

function assemble_dirichlet_datum(
  ::NTuple{3,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)

  if nonlin_lift ≤ 1
    return interpolate_dirichlet(x -> one(VectorValue(FEMInfo.D, Float)),
      FEMSpace.V)
  else
    return interpolate_dirichlet(Param.g(t), FEMSpace.V)
  end

end

function assemble_dirichlet_datum(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)

  function dirichlet_datum(t)
    if nonlin_lift ≤ 1
      return interpolate_dirichlet(Param.gₛ(t), FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
    end
  end

  dirichlet_datum

end

function assemble_dirichlet_datum(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_dirichlet_datum(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::FEMSpacePoissonUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_neumann_datum(t)

    if !FEMInfo.probl_nl["h"]
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    else
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    end

  end

  unsteady_neumann_datum

end

function assemble_neumann_datum(
  ::NTuple{2,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_neumann_datum(get_NTuple(1, Int), FEMSpace, FEMInfo, Param)

end

function assemble_neumann_datum(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesSteady,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["h"]
    hₛ(x) = x -> one(VectorValue(FEMInfo.D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_neumann_datum(t)
    if !FEMInfo.probl_nl["h"]
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    else
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    end
  end

  unsteady_neumann_datum

end

function assemble_neumann_datum(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_neumann_datum(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

#= function assemble_lifting(
  FEMSpace::FEMSpacePoissonSteady,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)
  gₕ = assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)

  if isodd(nonlin_lift)
    return assemble_vector(
      ∫(Param.α*(∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ)))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{Float}
  else
    return assemble_vector(
      ∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_lifting(
  FEMSpace::FEMSpacePoissonUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)
  gₕ(t) = assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)(t)

  function lifting_op(t)
    if isodd(nonlin_lift)
      return assemble_vector(
        ∫(Param.α(t)*(∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ(t))))*FEMSpace.dΩ,FEMSpace.V₀)
    else
      return assemble_vector(
        ∫(Param.αₛ(t)*∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ(t)))*FEMSpace.dΩ,FEMSpace.V₀)
    end
  end

  lifting_op

end

function assemble_lifting(
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)
  gₕ(t) = assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)(t)

  function lifting_op(t)
    if isodd(nonlin_lift)
      return assemble_vector(
        ∫(Param.α(t)*(∇(FEMSpace.ϕᵥ) ⊙ ∇(gₕ(t))))*FEMSpace.dΩ,FEMSpace.V₀)
    else
      return assemble_vector(
        ∫(Param.αₛ(t)*∇(FEMSpace.ϕᵥ) ⊙ ∇(gₕ(t)))*FEMSpace.dΩ,FEMSpace.V₀)
    end
  end

  lifting_op

end

function assemble_continuity_lifting(
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  gₕ(t) = assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)(t)

  lifting_op(t) =
    assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(gₕ(t))))*FEMSpace.dΩ,FEMSpace.Q₀)

end =#

function assemble_L²_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesUnsteady)

  assemble_matrix(∫(FEMSpace.ψᵧ*FEMSpace.ψₚ)*FEMSpace.dΩ,
    FEMSpace.Q, FEMSpace.Q₀)::SparseMatrixCSC{Float, Int}

end

function assemble_L²_norm_matrix(
  ::NTuple{4,Int},
  FEMSpace::FEMSpaceStokesUnsteady)

  assemble_L²_norm_matrix(get_NTuple(3, Int), FEMSpace)

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::FEMSpacePoissonSteady)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::FEMSpacePoissonUnsteady)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FEMSpaceStokesUnsteady)

  assemble_H¹_norm_matrix(get_NTuple(1, Int), FEMSpace)

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesSteady)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FEMSpaceStokesUnsteady)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{4,Int},
  FEMSpace::FEMSpaceStokesUnsteady)

  assemble_H¹_norm_matrix(get_NTuple(3, Int), FEMSpace)

end


function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info,
  var::String)

  NT = FEMInfo.problem_id

  if var == "A"
    assemble_stiffness(NT,FEMSpace,FEMInfo,Param)
  elseif var == "B"
    assemble_advection(NT,FEMSpace,FEMInfo,Param)
  elseif var == "Bₚ"
    assemble_primal_op(NT,FEMSpace)
  elseif var == "C"
    assemble_convection(NT,FEMSpace,Param)
  elseif var == "D"
    assemble_reaction(NT,FEMSpace,FEMInfo,Param)
  elseif var == "F"
    assemble_forcing(NT,FEMSpace,FEMInfo,Param)
  elseif var == "G"
    assemble_dirichlet_datum(NT,FEMSpace,FEMInfo,Param)
  elseif var == "H"
    assemble_neumann_datum(NT,FEMSpace,FEMInfo,Param)
  #= elseif var == "L"
    assemble_lifting(FEMSpace,FEMInfo,Param)
  elseif var == "L_cont"
    assemble_continuity_lifting(FEMSpace,FEMInfo,Param) =#
  elseif var == "M"
    assemble_mass(NT,FEMSpace,FEMInfo,Param)
  elseif var == "Xᵘ"
    assemble_H¹_norm_matrix(NT,FEMSpace)
  elseif var == "Xᵖ"
    assemble_L²_norm_matrix(NT,FEMSpace)
  end

end
