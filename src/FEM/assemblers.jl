function assemble_mass(
  FEMSpace::FEMSpacePoissonSteady,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["M"]
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.m*FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_mass(
  FEMSpace::FEMSpacePoissonUnsteady,
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
  FEMSpace::FEMSpaceStokesUnsteady,
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

function assemble_stiffness(
  FEMSpace::FEMSpacePoissonSteady,
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
  FEMSpace::FEMSpacePoissonUnsteady,
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
  FEMSpace::FEMSpacePoissonUnsteady,
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

  return unsteady_stiffness

end

function assemble_stiffness(
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
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady)

  if !FEMInfo.probl_nl["f"] && !FEMInfo.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
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

function assemble_dirichlet_datum(
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo{T},
  Param::ParametricInfoSteady) where T

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)

  if nonlin_lift ≤ 1
    return interpolate_dirichlet(x->one(T), FEMSpace.V)
  else
    return interpolate_dirichlet(Param.g(t), FEMSpace.V)
  end

end

function assemble_dirichlet_datum(
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady) where T

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

function assemble_neumann_datum(
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo{T},
  Param::ParametricInfoSteady) where T

  if !FEMInfo.probl_nl["h"]
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{T}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{T}
  end

end

function assemble_neumann_datum(
  FEMSpace::FEMSpacePoissonUnsteady,
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady) where T

  function unsteady_neumann_datum(t)

    if !FEMInfo.probl_nl["h"]
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{T}
    else
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)::Vector{T}
    end

  end

  unsteady_neumann_datum

end

function assemble_neumann_datum(
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady) where T

  function unsteady_neumann_datum(t)
    if !FEMInfo.probl_nl["h"]
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{T}
    else
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)::Vector{T}
    end
  end

  unsteady_neumann_datum

end

function assemble_lifting(
  FEMSpace::FEMSpacePoissonSteady,
  FEMInfo::SteadyInfo{T},
  Param::ParametricInfoSteady) where T

  nonlin_lift = nonlinearity_lifting_op(FEMInfo)
  gₕ = assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)

  if isodd(nonlin_lift)
    return assemble_vector(
      ∫(Param.α*(∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ)))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{T}
  else
    return assemble_vector(
      ∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{T}
  end

end

function assemble_lifting(
  FEMSpace::FEMSpacePoissonUnsteady,
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady) where T

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
  FEMSpace::FEMSpaceStokesUnsteady{D,T},
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady) where {D,T}

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

function assemble_second_lifting(
  FEMSpace::FEMSpaceStokesUnsteady,
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady) where T

  gₕ(t) = assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)(t)

  lifting_op(t) =
    assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(gₕ(t))))*FEMSpace.dΩ,FEMSpace.Q₀)

end

function assemble_L²_norm_matrix(
  FEMSpace::FEMSpaceStokesUnsteady{D,T}) where {D,T}

  assemble_matrix(∫(FEMSpace.ψᵧ*FEMSpace.ψₚ)*FEMSpace.dΩ,
  FEMSpace.Q, FEMSpace.Q₀)::SparseMatrixCSC{T, Int64}

end

function assemble_L²₀_norm_matrix(
  FEMSpace₀::FEMSpaceStokesUnsteady{D,T}) where {D,T}

  assemble_matrix(∫(FEMSpace₀.ψᵧ*FEMSpace₀.ψₚ)*FEMSpace₀.dΩ,
  FEMSpace₀.Q, FEMSpace₀.Q₀)::SparseMatrixCSC{T, Int64}

end

function assemble_H¹_norm_matrix(
  FEMSpace::FEMSpacePoissonSteady{D,T}) where {D,T}

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{T, Int64}

end

function assemble_H¹_norm_matrix(
  FEMSpace::FEMSpacePoissonUnsteady{D,T}) where {D,T}

  Xᵘ(t) = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀))

  Xᵘ(0.0)::SparseMatrixCSC{T, Int64}

end

function assemble_H¹_norm_matrix(
  FEMSpace::FEMSpaceStokesUnsteady{D,T}) where {D,T}

  Xᵘ(t) = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀))

  Xᵘ(0.0)::SparseMatrixCSC{T, Int64}

end

function assemble_H¹₀_norm_matrix(
  FEMSpace₀::FEMSpacePoissonSteady{D,T}) where {D,T}

  (assemble_matrix(∫(∇(FEMSpace₀.ϕᵥ)⋅∇(FEMSpace₀.ϕᵤ))*FEMSpace₀.dΩ,
  FEMSpace₀.V, FEMSpace₀.V₀) +
  assemble_matrix(∫(FEMSpace₀.ϕᵥ*FEMSpace₀.ϕᵤ)*FEMSpace₀.dΩ,
  FEMSpace₀.V, FEMSpace₀.V₀))::SparseMatrixCSC{T, Int64}

end

function assemble_H¹₀_norm_matrix(
  FEMSpace₀::FEMSpacePoissonUnsteady{D,T}) where {D,T}

  Xᵘ₀(t) = (assemble_matrix(∫(∇(FEMSpace₀.ϕᵥ)⋅∇(FEMSpace₀.ϕᵤ(t)))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀) +
  assemble_matrix(∫(FEMSpace₀.ϕᵥ * FEMSpace₀.ϕᵤ(t))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀))

  return Xᵘ₀(0.0)::SparseMatrixCSC{T, Int64}

end

function assemble_H¹₀_norm_matrix(
  FEMSpace₀::FEMSpaceStokesUnsteady{D,T}) where {D,T}

  Xᵘ₀(t) = (assemble_matrix(∫(∇(FEMSpace₀.ϕᵥ)⊙∇(FEMSpace₀.ϕᵤ(t)))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀) +
  assemble_matrix(∫(FEMSpace₀.ϕᵥ⋅FEMSpace₀.ϕᵤ(t))*FEMSpace₀.dΩ,
  FEMSpace₀.V(t), FEMSpace₀.V₀))

  return Xᵘ₀(0.0)::SparseMatrixCSC{T, Int64}

end

function assemble_H¹_norm_matrix(
  FEMSpace::FEMSpaceStokesUnsteady{D,T}) where {D,T}

  Xᵘ(t) = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(t)))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(t))*FEMSpace.dΩ,
  FEMSpace.V(t), FEMSpace.V₀))

  return Xᵘ(0.0)::SparseMatrixCSC{T, Int64}

end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::ParametricInfo,
  var::String)

  if var == "A"
    assemble_stiffness(FEMSpace,FEMInfo,Param)
  elseif var == "B"
    assemble_primal_op(FEMSpace)
  elseif var == "C"
    assemble_convective_op(FEMSpace)
  elseif var == "F"
    assemble_forcing(FEMSpace,FEMInfo,Param)
  elseif var == "G"
    assemble_dirichlet_datum(FEMSpace,FEMInfo,Param)
  elseif var == "H"
    assemble_neumann_datum(FEMSpace,FEMInfo,Param)
  elseif var == "M"
    assemble_mass(FEMSpace,FEMInfo,Param)
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
