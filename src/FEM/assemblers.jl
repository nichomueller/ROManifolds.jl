function assemble_mass(FE_space::SteadyProblem, probl::SteadyInfo, param::ParametricSpecifics)

  if !probl.probl_nl["M"]
    return assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  else
    return assemble_matrix(∫(FE_space.ϕᵥ * (param.m * FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  end

end

function assemble_mass(FE_space::UnsteadyProblem, probl::UnsteadyInfo, param::ParametricSpecificsUnsteady)

  function unsteady_mass(t)
    if !probl.probl_nl["M"]
      return assemble_matrix(∫(FE_space.ϕᵥ * (param.mₛ * FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    else
      return assemble_matrix(∫(FE_space.ϕᵥ * (param.m(t) * FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    end
  end

  return unsteady_mass

end

function assemble_stiffness(FE_space::SteadyProblem, probl::SteadyInfo, param::ParametricSpecifics)

  if !probl.probl_nl["A"]
    A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  else
    A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α * ∇(FE_space.ϕᵤ))) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  end

  return A

end

function assemble_stiffness(FE_space::UnsteadyProblem, probl::UnsteadyInfo, param::ParametricSpecificsUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.αₛ * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    else
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    end
  end

  return unsteady_stiffness

end

function assemble_stiffness(FE_space::UnsteadyProblem, probl::UnsteadyInfo, param::ParametricSpecificsUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.αₛ * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    else
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    end
  end

  return unsteady_stiffness

end

function assemble_primal_opᵀ(FE_space::FEMProblem)

  return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ FE_space.ϕₚ) * FE_space.dΩ, FE_space.Q, FE_space.V₀)

end

function assemble_primal_op(FE_space::SteadyProblem)

  return assemble_matrix(∫(FE_space.ϕᵧ ⋅  ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.Q₀)

end

function assemble_primal_op(FE_space::UnsteadyProblem)

  return assemble_matrix(∫(FE_space.ϕᵧ ⋅  ∇(FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.Q₀)

end

function assemble_forcing(FE_space::SteadyProblem, probl::SteadyInfo, param::ParametricSpecifics)

  if !probl.probl_nl["f"] && !probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΩ, FE_space.V₀)
  else
    return assemble_vector(∫(FE_space.ϕᵥ * param.f) * FE_space.dΩ, FE_space.V₀)
  end

end

function assemble_forcing(FE_space::UnsteadyProblem, probl::UnsteadyInfo, param::ParametricSpecificsUnsteady)

  function unsteady_forcing(t)
    if !probl.probl_nl["f"]
      return assemble_vector(∫(FE_space.ϕᵥ * param.fₛ) * FE_space.dΩ, FE_space.V₀)
    else probl.probl_nl["f"]
      return assemble_vector(∫(FE_space.ϕᵥ * param.f(t)) * FE_space.dΩ, FE_space.V₀)
    end
  end

  return unsteady_forcing

end

function assemble_neumann_datum(FE_space::SteadyProblem, probl::SteadyInfo, param::ParametricSpecifics)

  if !probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΓn, FE_space.V₀)
  else
    return assemble_vector(∫(FE_space.ϕᵥ * param.h) * FE_space.dΓn, FE_space.V₀)
  end

end

function assemble_neumann_datum(FE_space::UnsteadyProblem, probl::UnsteadyInfo, param::ParametricSpecificsUnsteady)

  function unsteady_neumann_datum(t)
    if !probl.probl_nl["h"]
      return assemble_vector(∫(FE_space.ϕᵥ * param.hₛ) * FE_space.dΓn, FE_space.V₀)
    else
      return assemble_vector(∫(FE_space.ϕᵥ * param.h(t)) * FE_space.dΓn, FE_space.V₀)
    end
  end

  return unsteady_neumann_datum

end

function assemble_dirichlet_datum(FE_space::SteadyProblem, probl::SteadyInfo, param::ParametricSpecifics)

  if !probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΓd, FE_space.V₀)
  else
    return assemble_vector(∫(FE_space.ϕᵥ * param.g) * FE_space.dΓd, FE_space.V₀)
  end

end

function assemble_dirichlet_datum(FE_space::UnsteadyProblem, probl::UnsteadyInfo, param::ParametricSpecificsUnsteady)

  function unsteady_neumann_datum(t)
    if !probl.probl_nl["h"]
      return assemble_vector(∫(FE_space.ϕᵥ * param.gₛ) * FE_space.dΓd, FE_space.V₀)
    else
      return assemble_vector(∫(FE_space.ϕᵥ * param.g(t)) * FE_space.dΓd, FE_space.V₀)
    end
  end

  return unsteady_neumann_datum

end

function assemble_L2_norm_matrix(FE_space::FEMProblem)

  Xᵖ = assemble_matrix(∫(FE_space.ψᵧ * FE_space.ψₚ) * FE_space.dΩ, FE_space.Q, FE_space.Q₀)

  return Xᵖ

end

function assemble_L2_norm_matrix_nobcs(FE_space₀::FEMProblem)

  Xᵖ₀ = assemble_matrix(∫(FE_space₀.ψᵧ * FE_space₀.ψₚ) * FE_space₀.dΩ, FE_space₀.Q, FE_space₀.Q₀)

  return Xᵖ₀

end

function assemble_H1_norm_matrix(FE_space::SteadyProblem)

  Xᵘ = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀) +
  assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ) * FE_space.dΩ, FE_space.V, FE_space.V₀)

  return Xᵘ

end

function assemble_H1_norm_matrix(FE_space::UnsteadyProblem)

  Xᵘ(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀) +
  assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ(t)) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)

  return Xᵘ(0.0)

end

function assemble_H1_norm_matrix_nobcs(FE_space₀::SteadyProblem)

  Xᵘ₀ = assemble_matrix(∫(∇(FE_space₀.ϕᵥ) ⋅ ∇(FE_space₀.ϕᵤ)) * FE_space₀.dΩ, FE_space₀.V, FE_space₀.V₀) +
  assemble_matrix(∫(FE_space₀.ϕᵥ * FE_space₀.ϕᵤ) * FE_space₀.dΩ, FE_space₀.V, FE_space₀.V₀)

  return Xᵘ₀

end

function assemble_H1_norm_matrix_nobcs(FE_space₀::UnsteadyProblem)

  Xᵘ₀(t) = assemble_matrix(∫(∇(FE_space₀.ϕᵥ) ⋅ ∇(FE_space₀.ϕᵤ(t))) * FE_space₀.dΩ, FE_space₀.V(t), FE_space₀.V₀) +
  assemble_matrix(∫(FE_space₀.ϕᵥ * FE_space₀.ϕᵤ(t)) * FE_space₀.dΩ, FE_space₀.V(t), FE_space₀.V₀)

  return Xᵘ₀(0.0)

end
