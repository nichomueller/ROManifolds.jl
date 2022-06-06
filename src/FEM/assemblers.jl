function assemble_mass(
  FESpace::FESpacePoissonSteady,
  probl::SteadyInfo,
  Param::ParametricSpecifics)

  if !probl.probl_nl["M"]
    assemble_matrix(∫(FESpace.ϕᵥ*FESpace.ϕᵤ)*FESpace.dΩ,
    FESpace.V, FESpace.V₀)
  else
    assemble_matrix(∫(FESpace.ϕᵥ*(Param.m*FESpace.ϕᵤ))*FESpace.dΩ,
    FESpace.V, FESpace.V₀)
  end

end

function assemble_mass(
  FESpace::FESpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_mass(t)
    if !probl.probl_nl["M"]
      assemble_matrix(∫(FESpace.ϕᵥ*(Param.mₛ*FESpace.ϕᵤ(t)))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    else
      assemble_matrix(∫(FESpace.ϕᵥ*(Param.m(t)*FESpace.ϕᵤ(t)))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    end
  end

  return unsteady_mass

end

function assemble_mass(
  FESpace::FESpaceStokesUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_mass(t)
    if !probl.probl_nl["M"]
      assemble_matrix(∫(FESpace.ϕᵥ⋅(Param.mₛ*FESpace.ϕᵤ(t)))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    else
      assemble_matrix(∫(FESpace.ϕᵥ⋅(Param.m(t)*FESpace.ϕᵤ(t)))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    end
  end

  return unsteady_mass

end

function assemble_stiffness(
  FESpace::FESpacePoissonSteady,
  probl::SteadyInfo,
  Param::ParametricSpecifics)

  if !probl.probl_nl["A"]
    assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅∇(FESpace.ϕᵤ))*FESpace.dΩ,
    FESpace.V, FESpace.V₀)
  else
    assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅(Param.α*∇(FESpace.ϕᵤ)))*FESpace.dΩ,
    FESpace.V, FESpace.V₀)
  end

end

function assemble_stiffness(
  FESpace::FESpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅(Param.αₛ*∇(FESpace.ϕᵤ(t))))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    else
      assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅(Param.α(t)*∇(FESpace.ϕᵤ(t))))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    end
  end

  unsteady_stiffness

end

function assemble_stiffness(
  FESpace::FESpacePoissonUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅(Param.αₛ*∇(FESpace.ϕᵤ(t))))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    else
      assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅(Param.α(t)*∇(FESpace.ϕᵤ(t))))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    end
  end

  return unsteady_stiffness

end

function assemble_stiffness(
  FESpace::FESpaceStokesUnsteady,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      assemble_matrix(∫(∇(FESpace.ϕᵥ)⊙(Param.αₛ*∇(FESpace.ϕᵤ(t))))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    else
      assemble_matrix(∫(∇(FESpace.ϕᵥ)⊙(Param.α(t)*∇(FESpace.ϕᵤ(t))))*FESpace.dΩ,
      FESpace.V(t), FESpace.V₀)
    end
  end

  unsteady_stiffness

end

function assemble_primal_opᵀ(FESpace::FEMProblem)

  assemble_matrix(∫(∇⋅(FESpace.ϕᵥ)*FESpace.ψₚ)*FESpace.dΩ,
  FESpace.Q, FESpace.V₀)

end

function assemble_primal_op(FESpace::SteadyProblem)

  ssemble_matrix(∫(FESpace.ψᵧ*∇⋅(FESpace.ϕᵤ))*FESpace.dΩ,
  FESpace.V, FESpace.Q₀)

end

function assemble_primal_op(FESpace::UnsteadyProblem)

  function unsteady_primal_form(t)
    assemble_matrix(∫(FESpace.ψᵧ*(∇⋅(FESpace.ϕᵤ(t))))*FESpace.dΩ,
    FESpace.V(t), FESpace.Q₀)
  end

  unsteady_primal_form

end

function assemble_forcing(
  FESpace::SteadyProblem,
  probl::SteadyInfo,
  Param::ParametricSpecifics)

  if !probl.probl_nl["f"] && !probl.probl_nl["h"]
    assemble_vector(∫(FESpace.ϕᵥ)*FESpace.dΩ, FESpace.V₀)
  else
    assemble_vector(∫(FESpace.ϕᵥ*Param.f)*FESpace.dΩ, FESpace.V₀)
  end

end

function assemble_forcing(
  FESpace::UnsteadyProblem,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_forcing(t)
    if !probl.probl_nl["f"]
      assemble_vector(∫(FESpace.ϕᵥ*Param.fₛ)*FESpace.dΩ, FESpace.V₀)
    else probl.probl_nl["f"]
      assemble_vector(∫(FESpace.ϕᵥ*Param.f(t))*FESpace.dΩ, FESpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_neumann_datum(
  FESpace::SteadyProblem,
  probl::SteadyInfo,
  Param::ParametricSpecifics)

  if !isnothing(FESpace.dΓn)
    if !probl.probl_nl["h"]
      assemble_vector(∫(FESpace.ϕᵥ)*FESpace.dΓn, FESpace.V₀)
    else
      assemble_vector(∫(FESpace.ϕᵥ*Param.h)*FESpace.dΓn, FESpace.V₀)
    end
  else
    zeros(num_free_dofs(FESpace.V₀))
  end

end

function assemble_neumann_datum(
  FESpace::UnsteadyProblem,
  probl::UnsteadyInfo,
  Param::ParametricSpecificsUnsteady)

  function unsteady_neumann_datum(t)
    if !isnothing(FESpace.dΓn)
      if !probl.probl_nl["h"]
        assemble_vector(∫(FESpace.ϕᵥ*Param.hₛ)*FESpace.dΓn, FESpace.V₀)
      else
        assemble_vector(∫(FESpace.ϕᵥ*Param.h(t))*FESpace.dΓn, FESpace.V₀)
      end
    else
      zeros(num_free_dofs(FESpace.V₀))
    end
  end

  unsteady_neumann_datum

end

function assemble_dirichlet_datum(
  FESpace::SteadyProblem,
  Param::ParametricSpecifics)

  if !isnothing(FESpace.dΓd)
    assemble_vector(∫(FESpace.ϕᵥ*Param.g)*FESpace.dΓd, FESpace.V₀)
  else
    zeros(num_free_dofs(FESpace.V₀))
  end

end

function assemble_dirichlet_datum(
  FESpace::UnsteadyProblem,
  Param::ParametricSpecificsUnsteady)

  function unsteady_dirichlet_datum(t)
    if !isnothing(FESpace.dΓd)
      assemble_vector(∫(FESpace.ϕᵥ*Param.g(t))*FESpace.dΓd, FESpace.V₀)
    else
      zeros(num_free_dofs(FESpace.V₀))
    end
  end

  return unsteady_dirichlet_datum

end

function assemble_L²_norm_matrix(FESpace::FESpaceStokesUnsteady)

  assemble_matrix(∫(FESpace.ψᵧ*FESpace.ψₚ)*FESpace.dΩ,
  FESpace.Q, FESpace.Q₀)

end

function assemble_L²₀_norm_matrix(FESpace₀::FESpaceStokesUnsteady)

  assemble_matrix(∫(FESpace₀.ψᵧ*FESpace₀.ψₚ)*FESpace₀.dΩ,
  FESpace₀.Q, FESpace₀.Q₀)

end

function assemble_H¹_norm_matrix(FESpace::FESpacePoissonSteady)

  (assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅∇(FESpace.ϕᵤ))*FESpace.dΩ,
  FESpace.V, FESpace.V₀) +
  assemble_matrix(∫(FESpace.ϕᵥ*FESpace.ϕᵤ)*FESpace.dΩ,
  FESpace.V, FESpace.V₀))

end

function assemble_H¹_norm_matrix(FESpace::FESpacePoissonUnsteady)

  Xᵘ(t) = (assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅∇(FESpace.ϕᵤ(t)))*FESpace.dΩ,
  FESpace.V(t), FESpace.V₀) +
  assemble_matrix(∫(FESpace.ϕᵥ*FESpace.ϕᵤ(t))*FESpace.dΩ,
  FESpace.V(t), FESpace.V₀))

  Xᵘ(0.0)

end

function assemble_H¹_norm_matrix(FESpace::FESpaceStokesUnsteady)

  Xᵘ(t) = (assemble_matrix(∫(∇(FESpace.ϕᵥ)⊙∇(FESpace.ϕᵤ(t)))*FESpace.dΩ,
  FESpace.V(t), FESpace.V₀) +
  assemble_matrix(∫(FESpace.ϕᵥ⋅FESpace.ϕᵤ(t))*FESpace.dΩ,
  FESpace.V(t), FESpace.V₀))

  return Xᵘ(0.0)

end

function assemble_H¹₀_norm_matrix(FESpace₀::FESpacePoissonSteady)

  (assemble_matrix(∫(∇(FESpace₀.ϕᵥ)⋅∇(FESpace₀.ϕᵤ))*FESpace₀.dΩ,
  FESpace₀.V, FESpace₀.V₀) +
  assemble_matrix(∫(FESpace₀.ϕᵥ*FESpace₀.ϕᵤ)*FESpace₀.dΩ,
  FESpace₀.V, FESpace₀.V₀))

end

function assemble_H¹₀_norm_matrix(FESpace₀::FESpacePoissonUnsteady)

  Xᵘ₀(t) = (assemble_matrix(∫(∇(FESpace₀.ϕᵥ)⋅∇(FESpace₀.ϕᵤ(t)))*FESpace₀.dΩ,
  FESpace₀.V(t), FESpace₀.V₀) +
  assemble_matrix(∫(FESpace₀.ϕᵥ * FESpace₀.ϕᵤ(t))*FESpace₀.dΩ,
  FESpace₀.V(t), FESpace₀.V₀))

  return Xᵘ₀(0.0)

end

function assemble_H¹_norm_matrix_nobcs(FESpace₀::FESpaceStokesUnsteady)

  Xᵘ₀(t) = (assemble_matrix(∫(∇(FESpace₀.ϕᵥ)⊙∇(FESpace₀.ϕᵤ(t)))*FESpace₀.dΩ,
  FESpace₀.V(t), FESpace₀.V₀) +
  assemble_matrix(∫(FESpace₀.ϕᵥ⋅FESpace₀.ϕᵤ(t))*FESpace₀.dΩ,
  FESpace₀.V(t), FESpace₀.V₀))

  return Xᵘ₀(0.0)

end

function assemble_FEM_structure(
  FESpace::SteadyProblem,
  probl::SteadyInfo,
  Param::ParametricSpecifics,
  var::String)
  if var == "A"
    assemble_stiffness(FESpace,probl,Param)
  elseif var == "B"
    assemble_primal_op(FESpace)
  elseif var == "C"
    assemble_convective_op(FESpace)
  elseif var == "F"
    assemble_forcing(FESpace,probl,Param)
  elseif var == "G"
    assemble_dirichlet_datum(FESpace,Param)
  elseif var == "H"
    assemble_neumann_datum(FESpace,probl,Param)
  elseif var == "M"
    assemble_mass(FESpace,probl,Param)
  elseif var == "Xᵘ"
    assemble_H¹_norm_matrix(FESpace)
  elseif var == "Xᵖ"
    assemble_L²_norm_matrix(FESpace)
  elseif var == "Xᵘ₀"
    assemble_H¹₀_norm_matrix(FESpace)
  elseif var == "Xᵖ₀"
    assemble_L²₀_norm_matrix(FESpace)
  end
end
