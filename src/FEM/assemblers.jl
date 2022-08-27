function assemble_mass(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  Param::UnsteadyParametricInfo)

  assemble_mass(get_NTuple(1, Int), FEMSpace, FEMInfo, Param)

end

function assemble_mass(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  Param::UnsteadyParametricInfo)

  assemble_mass(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

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
  Param::UnsteadyParametricInfo)

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
  FEMSpace::SteadyProblem,
  ::SteadyInfo,
  Param::SteadyParametricInfo)

  ####### USING PECHLET STABILIZATION OF α #######
  α_stab = get_α_stab(FEMSpace, Param)
  ################################################

  assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(α_stab*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)

end

function assemble_stiffness(
  ::NTuple{2,Int},
  FEMSpace::UnsteadyProblem,
  ::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

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
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  FEMSpace::SteadyProblem,
  ::SteadyInfo,
  Param::SteadyParametricInfo)

  assemble_matrix(∫(FEMSpace.ϕᵥ * (Param.b⋅∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.V₀)

end

function assemble_advection(
  ::NTuple{2,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  FEMSpace::SteadyProblem,
  Param::SteadyParametricInfo)

  C(u) = Param.Re * assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    ((∇FEMSpace.ϕᵤ')⋅u) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  C

end

function assemble_convection(
  ::NTuple{4,Int},
  FEMSpace::UnsteadyProblem,
  Param::UnsteadyParametricInfo)

  C(u,t) = Param.Re * assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    ((∇FEMSpace.ϕᵤ')⋅u) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  C

end

function assemble_reaction(
  ::NTuple{2,Int},
  FEMSpace::SteadyProblem,
  ::SteadyInfo,
  Param::SteadyParametricInfo)

  assemble_matrix(∫(FEMSpace.ϕᵥ * FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)

end

function assemble_reaction(
  ::NTuple{2,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  Param::SteadyParametricInfo)

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
  Param::UnsteadyParametricInfo)

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
  Param::SteadyParametricInfo)

  if !FEMInfo.probl_nl["f"] && !FEMInfo.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  FEMSpace::SteadyProblem{D},
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo) where D

  if !FEMInfo.probl_nl["f"]
    fₛ(x) = x -> one(VectorValue(FEMInfo.D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

  if !FEMInfo.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

  if !FEMInfo.probl_nl["h"]
    hₛ(x) = x -> one(VectorValue(FEMInfo.D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

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

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

  if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    g_α = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    g_α = interpolate_dirichlet(Param.α * Param.g, FEMSpace.V)
  end

  assemble_vector(
    ∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(g_α))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

  δtθ = FEMInfo.δt*FEMInfo.θ

  function g_m(t)
    if !FEMInfo.probl_nl["M"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.mₛ * Param.gₛ, FEMSpace.V(t))
    else
      dg(x,t::Real) = ∂t(g)(x,t)
      dg(t::Real) = x -> dg(x,t)
      return interpolate_dirichlet(Param.m(t) * Param.dg(t), FEMSpace.V(t))
    end
  end

  function g_α(t)
    if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.αₛ * Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.α(t) * Param.g(t), FEMSpace.V(t))
    end
  end

  L(t) = (assemble_vector(∫(FEMSpace.ϕᵥ * g_m(t))*FEMSpace.dΩ,FEMSpace.V₀) / δtθ +
    assemble_vector(∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(g_α(t)))*FEMSpace.dΩ,FEMSpace.V₀))

  L

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

  if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    g_α = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    α_stab = get_α_stab(FEMSpace, Param)
    g_α = interpolate_dirichlet(α_stab * Param.g, FEMSpace.V)
  end

  if !FEMInfo.probl_nl["g"]
    g_b = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    g_b = interpolate_dirichlet(Param.g, FEMSpace.V)
  end

  if !FEMInfo.probl_nl["R"] && !FEMInfo.probl_nl["g"]
    g_r = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    g_r = interpolate_dirichlet(Param.σ * Param.g, FEMSpace.V)
  end

  (assemble_vector(∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(g_α))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(FEMSpace.ϕᵥ * (Param.b ⋅ ∇(g_b)))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(FEMSpace.ϕᵥ * g_r)*FEMSpace.dΩ,FEMSpace.V₀))::Vector{Float}

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

  δtθ = FEMInfo.δt*FEMInfo.θ

  function g_m(t)
    if !FEMInfo.probl_nl["M"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.mₛ * Param.gₛ, FEMSpace.V(t))
    else
      dg(x,t::Real) = ∂t(g)(x,t)
      dg(t::Real) = x -> dg(x,t)
      return interpolate_dirichlet(Param.m(t) * Param.dg(t), FEMSpace.V(t))
    end
  end

  function g_α(t)
    if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.αₛ * Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.α(t) * Param.g(t), FEMSpace.V(t))
    end
  end

  function g_b(t)
    if !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
    end
  end

  function g_r(t)
    if !FEMInfo.probl_nl["r"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.σₛ * Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.σ(t) * Param.g(t), FEMSpace.V(t))
    end
  end

  L(t) = (assemble_vector(∫(FEMSpace.ϕᵥ * g_m(t))*FEMSpace.dΩ,FEMSpace.V₀) / δtθ +
    assemble_vector(∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(g_α(t)))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(FEMSpace.ϕᵥ * (Param.b(t) ⋅ ∇(g_b(t))))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(FEMSpace.ϕᵥ * g_r(t))*FEMSpace.dΩ,FEMSpace.V₀))

  L

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

  if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    g_α = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    g_α = interpolate_dirichlet(Param.α * Param.g, FEMSpace.V)
  end

  if !FEMInfo.probl_nl["g"]
    g_b = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    g_b = interpolate_dirichlet(Param.g, FEMSpace.V)
  end

  vcat(assemble_vector(∫(∇(FEMSpace.ϕᵥ) ⊙ ∇(g_α))*FEMSpace.dΩ,FEMSpace.V₀),
    assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(g_b)))*FEMSpace.dΩ,FEMSpace.Q₀))::Vector{Float}

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

  function g_m(t)
    if !FEMInfo.probl_nl["M"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.mₛ * Param.gₛ, FEMSpace.V(t))
    else
      dg(x,t::Real) = ∂t(g)(x,t)
      dg(t::Real) = x -> dg(x,t)
      return interpolate_dirichlet(Param.m(t) * Param.dg(t), FEMSpace.V(t))
    end
  end

  function g_α(t)
    if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.αₛ * Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.α(t) * Param.g(t), FEMSpace.V(t))
    end
  end

  function g_b(t)
    if !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
    end
  end

  L(t) = vcat(assemble_vector(∫(FEMSpace.ϕᵥ ⋅ g_m(t))*FEMSpace.dΩ,FEMSpace.V₀) / δtθ +
    assemble_vector(∫(∇(FEMSpace.ϕᵥ) ⊙ ∇(g_α(t)))*FEMSpace.dΩ,FEMSpace.V₀),
    assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(g_b(t))))*FEMSpace.dΩ,FEMSpace.Q₀))

  L

end

function assemble_lifting(
  ::NTuple{4,Int},
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::SteadyParametricInfo)

  C(u) = Param.Re * assemble_vector(∫( FEMSpace.ϕᵥ ⊙
    ((FEMSpace.ϕᵥ')⋅u) )*FEMSpace.dΩ, FEMSpace.V₀)

  if !FEMInfo.probl_nl["g"]
    g = interpolate_dirichlet(x->1., FEMSpace.V)
  else
    g = interpolate_dirichlet(Param.g, FEMSpace.V)
  end

  (assemble_lifting(get_NTuple(3, Int), FEMSpace, FEMInfo, Param) +
    vcat(C(g), zeros(FEMSpace.Nₛᵖ)))::Vector{Float}

end

function assemble_lifting(
  ::NTuple{4,Int},
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::UnsteadyParametricInfo)

  C(u,t) = Param.Re * assemble_vector(∫( FEMSpace.ϕᵥ ⊙
    ((FEMSpace.ϕᵥ')⋅u(t)) )*FEMSpace.dΩ, FEMSpace.V₀)

  function g(t)
    if !FEMInfo.probl_nl["g"]
      return interpolate_dirichlet(Param.gₛ, FEMSpace.V(t))
    else
      return interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
    end
  end

  L(t) = (assemble_lifting(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)(t) +
    vcat(C(g(t),t), zeros(FEMSpace.Nₛᵖ)))::Vector{Float}

  L

end

function assemble_L²_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FEMProblem)

  assemble_matrix(∫(FEMSpace.ψᵧ*FEMSpace.ψₚ)*FEMSpace.dΩ,
    FEMSpace.Q, FEMSpace.Q₀)::SparseMatrixCSC{Float, Int}

end

function assemble_L²_norm_matrix(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem)

  assemble_L²_norm_matrix(get_NTuple(3, Int), FEMSpace)

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::SteadyProblem)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::UnsteadyProblem)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FEMProblem)

  assemble_H¹_norm_matrix(get_NTuple(1, Int), FEMSpace)

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::SteadyProblem)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::UnsteadyProblem)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem)

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
  elseif var == "L"
    assemble_lifting(NT,FEMSpace,FEMInfo,Param)
  elseif var == "M"
    assemble_mass(NT,FEMSpace,FEMInfo,Param)
  elseif var == "Xᵘ"
    assemble_H¹_norm_matrix(NT,FEMSpace)
  elseif var == "Xᵖ"
    assemble_L²_norm_matrix(NT,FEMSpace)
  end

end
