function assemble_mass(
  FEMSpace::FEMSpacePoissonSteady,
  probl::SteadyInfo,
  Param::ParametricInfoSteady)

  if !probl.probl_nl["M"]
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.m*FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_mass(
  FEMSpace::FEMSpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_mass(t)
    if !probl.probl_nl["M"]
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
  FEMSpace::FEMSpaceStokesUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_mass(t)
    if !probl.probl_nl["M"]
      assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Param.mₛ*FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Param.m(t)*FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  return unsteady_mass

end

function assemble_stiffness(
  FEMSpace::FEMSpacePoissonSteady,
  probl::SteadyInfo,
  Param::ParametricInfoSteady)

  if !probl.probl_nl["A"]
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  FEMSpace::FEMSpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
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
  FEMSpace::FEMSpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.αₛ*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  return unsteady_stiffness

end

function assemble_stiffness(
  FEMSpace::FEMSpaceStokesUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.αₛ*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    else
      assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
      FEMSpace.V(t), FEMSpace.V₀)
    end
  end

  unsteady_stiffness

end

function assemble_primal_op(FEMSpace::SteadyProblem)

  assemble_matrix(∫(FEMSpace.ψᵧ*∇⋅(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.Q₀)

end

function assemble_primal_op(FEMSpace::UnsteadyProblem)

  function unsteady_primal_form(t)
    assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
    FEMSpace.V(t), FEMSpace.Q₀)
  end

  unsteady_primal_form

end

function assemble_forcing(
  FEMSpace::SteadyProblem,
  probl::SteadyInfo,
  Param::ParametricInfoSteady)

  if !probl.probl_nl["f"] && !probl.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  FEMSpace::FEMSpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_forcing(t)
    if !probl.probl_nl["f"]
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
    else probl.probl_nl["f"]
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.f(t))*FEMSpace.dΩ, FEMSpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_forcing(
  FEMSpace::FEMSpaceStokesUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_forcing(t)
    if !probl.probl_nl["f"]
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
    else probl.probl_nl["f"]
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f(t))*FEMSpace.dΩ, FEMSpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_neumann_datum(
  FEMSpace::SteadyProblem,
  probl::SteadyInfo,
  Param::ParametricInfoSteady)

  if !isnothing(FEMSpace.dΓn)
    if !probl.probl_nl["h"]
      assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)
    else
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)
    end
  else
    zeros(num_free_dofs(FEMSpace.V₀))
  end

end

function assemble_neumann_datum(
  FEMSpace::FEMSpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_neumann_datum(t)
    if !isnothing(FEMSpace.dΓn)
      if !probl.probl_nl["h"]
        assemble_vector(∫(FEMSpace.ϕᵥ*Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)
      else
        assemble_vector(∫(FEMSpace.ϕᵥ*Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)
      end
    else
      zeros(num_free_dofs(FEMSpace.V₀))
    end
  end

  unsteady_neumann_datum

end

function assemble_neumann_datum(
  FEMSpace::FEMSpaceStokesUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricInfoUnsteady)

  function unsteady_neumann_datum(t)
    if !isnothing(FEMSpace.dΓn)
      if !probl.probl_nl["h"]
        assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)
      else
        assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)
      end
    else
      zeros(num_free_dofs(FEMSpace.V₀))
    end
  end

  unsteady_neumann_datum

end

function assemble_dirichlet_datum(
  FEMSpace::SteadyProblem,
  Param::ParametricInfoSteady)

  if !isnothing(FEMSpace.dΓd)
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.g)*FEMSpace.dΓd, FEMSpace.V₀)
  else
    zeros(num_free_dofs(FEMSpace.V₀))
  end

end

function assemble_dirichlet_datum(
  FEMSpace::UnsteadyProblem,
  Param::ParametricInfoUnsteady)

  function unsteady_dirichlet_datum(t)
    if !isnothing(FEMSpace.dΓd)
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.g(t))*FEMSpace.dΓd, FEMSpace.V₀)
    else
      zeros(num_free_dofs(FEMSpace.V₀))
    end
  end

  return unsteady_dirichlet_datum

end

function assemble_L²_norm_matrix(FEMSpace::FEMSpaceStokesUnsteady)

  assemble_matrix(∫(FEMSpace.ψᵧ*FEMSpace.ψₚ)*FEMSpace.dΩ,
  FEMSpace.Q, FEMSpace.Q₀)

end

function assemble_L²₀_norm_matrix(FEMSpace₀::FEMSpaceStokesUnsteady)

  assemble_matrix(∫(FEMSpace₀.ψᵧ*FEMSpace₀.ψₚ)*FEMSpace₀.dΩ,
  FEMSpace₀.Q, FEMSpace₀.Q₀)

end

function assemble_H¹_norm_matrix(FEMSpace::FEMSpacePoissonSteady)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))

end

function assemble_H¹_norm_matrix(FEMSpace::FEMSpacePoissonUnsteady)

  Xᵘ(t) = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀))

  Xᵘ(0.0)

end

function assemble_H¹_norm_matrix(FEMSpace::FEMSpaceStokesUnsteady)

  Xᵘ(t) = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀))

  return Xᵘ(0.0)

end

function assemble_H¹₀_norm_matrix(FEMSpace₀::FEMSpacePoissonSteady)

  (assemble_matrix(∫(∇(FEMSpace₀.ϕᵥ)⋅∇(FEMSpace₀.ϕᵤ))*FEMSpace₀.dΩ,
  FEMSpace₀.V, FEMSpace₀.V₀) +
  assemble_matrix(∫(FEMSpace₀.ϕᵥ*FEMSpace₀.ϕᵤ)*FEMSpace₀.dΩ,
  FEMSpace₀.V, FEMSpace₀.V₀))

end

function assemble_H¹₀_norm_matrix(FEMSpace₀::FEMSpacePoissonUnsteady)

  Xᵘ₀(t) = (assemble_matrix(∫(∇(FEMSpace₀.ϕᵥ)⋅∇(FEMSpace₀.ϕᵤ(t)))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀) +
  assemble_matrix(∫(FEMSpace₀.ϕᵥ * FEMSpace₀.ϕᵤ(t))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀))

  return Xᵘ₀(0.0)

end

function assemble_H¹₀_norm_matrix(FEMSpace₀::FEMSpaceStokesUnsteady)

  Xᵘ₀(t) = (assemble_matrix(∫(∇(FEMSpace₀.ϕᵥ)⊙∇(FEMSpace₀.ϕᵤ(t)))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀) +
  assemble_matrix(∫(FEMSpace₀.ϕᵥ⋅FEMSpace₀.ϕᵤ(t))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀))

  return Xᵘ₀(0.0)

end

function assemble_H¹_norm_matrix_nobcs(FEMSpace::FEMSpaceStokesUnsteady)

  Xᵘ(t) = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀))

  return Xᵘ(0.0)

end

function assemble_lifting(FEMSpace::SteadyProblem, Param::ParametricInfoSteady)

  Gₕ = zeros(FEMSpace.Nₛᵘ,1)
  if !isnothing(FEMSpace.dΓd)
    gₕ = interpolate_everywhere(Param.g, FEMSpace.V)
    Gₕ = get_free_dof_values(gₕ)
  end

  Gₕ

end

function assemble_lifting(
  FEMSpace::FEMSpacePoissonUnsteady, probl::ProblemInfoUnsteady,
  Param::ParametricInfoUnsteady)

  Gₕ = zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  if !isnothing(FEMSpace.dΓd)
    gₕ(t) = interpolate_everywhere(Param.g(t), FEMSpace.V(t))
    for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
      Gₕ[:, i] = get_free_dof_values(gₕ(tᵢ))
    end
  end

  Gₕ

end

function assemble_lifting(
  FEMSpace::FEMSpaceStokesUnsteady,
  ::ProblemInfoUnsteady,
  Param::ParametricInfoUnsteady)

  gₕ(t) = interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
  function R₁(t)
    if !isnothing(FEMSpace.dΓd)
      return assemble_vector(
        ∫(Param.α(t)*(∇(FEMSpace.ϕᵥ) ⊙ ∇(gₕ(t))))*FEMSpace.dΩ,FEMSpace.V₀)
    else
      return zeros(FEMSpace.Nₛᵘ)
    end
  end
  function R₂(t)
    if !isnothing(FEMSpace.dΓd)
      return assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(gₕ(t))))*FEMSpace.dΩ,FEMSpace.Q₀)
    else
      return zeros(FEMSpace.Nₛᵖ)
    end
  end
  _->(R₁,R₂)
end

function assemble_FEM_structure(
  FEMSpace::SteadyProblem,
  probl::SteadyInfo,
  Param::ParametricInfoSteady,
  var::String)
  if var == "A"
    assemble_stiffness(FEMSpace,probl,Param)
  elseif var == "B"
    assemble_primal_op(FEMSpace)
  elseif var == "C"
    assemble_convective_op(FEMSpace)
  elseif var == "F"
    assemble_forcing(FEMSpace,probl,Param)
  elseif var == "G"
    assemble_dirichlet_datum(FEMSpace,Param)
  elseif var == "H"
    assemble_neumann_datum(FEMSpace,probl,Param)
  elseif var == "M"
    assemble_mass(FEMSpace,probl,Param)
  elseif var == "Xᵘ"
    assemble_H¹_norm_matrix(FEMSpace)
  elseif var == "Xᵖ"
    assemble_L²_norm_matrix(FEMSpace)
  elseif var == "Xᵘ₀"
    assemble_H¹₀_norm_matrix(FEMSpace)
  elseif var == "Xᵖ₀"
    assemble_L²₀_norm_matrix(FEMSpace)
  end
end
