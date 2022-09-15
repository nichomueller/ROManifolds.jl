function assemble_mass(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_mass(t)
    if "M" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  assemble_mass(get_NTuple(1, Int), FEMSpace, FEMInfo, Param)

end

function assemble_mass(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_mass(t)
    if "M" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  assemble_mass(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  if "A" ∉ FEMInfo.probl_nl
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_stiffness(t)
    if "A" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemS,
  ::InfoS,
  Param::SteadyParametricInfo)

  ####### USING PECHLET STABILIZATION OF α #######
  α_stab = get_α_stab(FEMSpace, Param)
  ################################################

  assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(α_stab*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)

end

function assemble_stiffness(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  ::InfoST,
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
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  if "A" ∉ FEMInfo.probl_nl
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_stiffness(t)
    if "A" ∉ FEMInfo.probl_nl
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

function assemble_B(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemS,
  ::Info,
  ::Info)

  assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.Q₀)

end

function assemble_B(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  ::Info,
  ::Info)

  function unsteady_primal_form(t)
    assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
    FEMSpace.V(t), FEMSpace.Q₀)
  end

  unsteady_primal_form

end

function assemble_B(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_B(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_B(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  ::InfoS,
  Param::SteadyParametricInfo)

  assemble_matrix(∫(FEMSpace.ϕᵥ * (Param.b⋅∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.V₀)

end

function assemble_B(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function advection(t)
    if "B" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemS,
  Param::SteadyParametricInfo)

  C(u) = Param.Re * assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    ((∇FEMSpace.ϕᵤ')⋅u) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  C

end

function assemble_convection(
  ::NTuple{4,Int},
  FEMSpace::FEMProblemST,
  Param::UnsteadyParametricInfo)

  C(u,t) = Param.Re * assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    ((∇FEMSpace.ϕᵤ')⋅u) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  C

end

function assemble_reaction(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  ::InfoS,
  Param::SteadyParametricInfo)

  assemble_matrix(∫(FEMSpace.ϕᵥ * FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)

end

function assemble_reaction(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function advection(t)
    if "R" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMSpaceADRS,
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
  FEMSpace::FEMSpaceADRST,
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
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  if "F" ∉ FEMInfo.probl_nl && "H" ∉ FEMInfo.probl_nl
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_forcing(t)
    if "F" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemS{D},
  FEMInfo::InfoS,
  Param::SteadyParametricInfo) where D

  if "F" ∉ FEMInfo.probl_nl
    fₛ = x -> one(VectorValue(FEMInfo.D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_forcing(t)
    if "F" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  if "H" ∉ FEMInfo.probl_nl
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_neumann_datum(t)

    if "H" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  if "H" ∉ FEMInfo.probl_nl
    hₛ = x -> one(VectorValue(FEMInfo.D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  function unsteady_neumann_datum(t)
    if "H" ∉ FEMInfo.probl_nl
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
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(
    ∫(Param.α * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  δtθ = FEMInfo.δt*FEMInfo.θ
  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L(t) = (assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ * dg(t))*FEMSpace.dΩ,FEMSpace.V₀) / δtθ +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀))

  L

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  g = define_g_FEM(FEMSpace, Param)

  (assemble_vector(∫(Param.α * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(FEMSpace.ϕᵥ * (Param.b ⋅ ∇(g)))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(Param.σ * FEMSpace.ϕᵥ * g)*FEMSpace.dΩ,FEMSpace.V₀))::Vector{Float}

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  δtθ = FEMInfo.δt*FEMInfo.θ
  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L(t) = δtθ * (assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ * dg(t))*FEMSpace.dΩ,FEMSpace.V₀) / δtθ +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(FEMSpace.ϕᵥ * (Param.b(t) ⋅ ∇(g(t))))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(Param.σ(t) * FEMSpace.ϕᵥ * g(t))*FEMSpace.dΩ,FEMSpace.V₀))

  L

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(∫(Param.α * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g))*FEMSpace.dΩ,
    FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  δtθ = FEMInfo.δt*FEMInfo.θ
  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L₁(t) = assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ ⋅ dg(t))*FEMSpace.dΩ,FEMSpace.V₀) / δtθ +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀)

  L₁

end

function assemble_lifting(
  ::NTuple{4,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  C(u) = Param.Re * assemble_vector(∫( FEMSpace.ϕᵥ ⊙
    ((FEMSpace.ϕᵥ')⋅u) )*FEMSpace.dΩ, FEMSpace.V₀)
  g = define_g_FEM(FEMSpace, Param)

  L₁ = assemble_lifting(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

  (L₁ + C(g))::Vector{Float}

end

function assemble_lifting(
  ::NTuple{4,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  C(u, t) = Param.Re * assemble_vector(∫( FEMSpace.ϕᵥ ⊙
    ((FEMSpace.ϕᵥ')⋅u(t)) )*FEMSpace.dΩ, FEMSpace.V₀)
  g = define_g_FEM(FEMSpace, Param)

  L₁ = assemble_lifting(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)
  L₁_new(t) = L₁(t) + C(g(t), t)

  L₁_new

end

function assemble_lifting_continuity(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::InfoS,
  Param::SteadyParametricInfo)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(∫(FEMSpace.ψᵧ * (∇⋅g))*FEMSpace.dΩ,FEMSpace.Q₀)::Vector{Float}

end

function assemble_lifting_continuity(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::InfoST,
  Param::UnsteadyParametricInfo)

  g = define_g_FEM(FEMSpace, Param)

  L₂(t) = assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(g(t))))*FEMSpace.dΩ,FEMSpace.Q₀)

  L₂

end

function assemble_lifting_continuity(
  ::NTuple{4,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_lifting_continuity(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

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
  FEMSpace::FEMProblemS)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST)

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
  FEMSpace::FEMProblemS)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST)

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
    assemble_B(NT,FEMSpace,FEMInfo,Param)
  elseif var == "C"
    assemble_convection(NT,FEMSpace,Param)
  elseif var == "D"
    assemble_reaction(NT,FEMSpace,FEMInfo,Param)
  elseif var == "F"
    assemble_forcing(NT,FEMSpace,FEMInfo,Param)
  elseif var == "H"
    assemble_neumann_datum(NT,FEMSpace,FEMInfo,Param)
  elseif var == "L"
    assemble_lifting(NT,FEMSpace,FEMInfo,Param)
  elseif var == "Lc"
    assemble_lifting_continuity(NT,FEMSpace,FEMInfo,Param)
  elseif var == "M"
    assemble_mass(NT,FEMSpace,FEMInfo,Param)
  elseif var == "Xᵘ"
    assemble_H¹_norm_matrix(NT,FEMSpace)
  elseif var == "Xᵖ"
    assemble_L²_norm_matrix(NT,FEMSpace)
  end

end
