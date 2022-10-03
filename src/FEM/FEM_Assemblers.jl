function assemble_mass(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  assemble_mass(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

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
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  if "A" ∉ FEMInfo.probl_nl
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  ::NTuple{3,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_stiffness(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_B(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  ::Info,
  ::Info)

  if "B" ∉ FEMInfo.probl_nl
    assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.Q₀)
  else
    assemble_matrix(∫(Param.b * FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.Q₀)
  end

end

function assemble_B(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  ::Info,
  ::Info)

  function unsteady_primal_form(t)
    if "B" ∉ FEMInfo.probl_nl
      assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ(t))))*FEMSpace.dΩ,
        FEMSpace.V(t), FEMSpace.Q₀)
    else
      assemble_matrix(∫(Param.b(t) * FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
        FEMSpace.V(t), FEMSpace.Q₀)
    end
  end

  unsteady_primal_form

end

function assemble_B(
  ::NTuple{3,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_B(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_convection(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemS)

  C(u) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(FEMSpace.ϕᵤ)'⋅u) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  C

end

function assemble_convection(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST)

  C(u,t) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(FEMSpace.ϕᵤ(t))'⋅u(t)) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  C

end

function assemble_swapped_convection(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemS)

  D(u) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(u)'⋅FEMSpace.ϕᵤ) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  D

end

function assemble_swapped_convection(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST)

  D(u,t) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(u(t))'⋅FEMSpace.ϕᵤ(t)) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  D

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  if "F" ∉ FEMInfo.probl_nl
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  FEMSpace::FEMProblemS{D},
  FEMInfo::FEMInfoS,
  Param::ParamInfoS) where D

  if "F" ∉ FEMInfo.probl_nl
    fₛ = x -> one(VectorValue(D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  ::NTuple{3,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_forcing(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  if "H" ∉ FEMInfo.probl_nl
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  FEMSpace::FEMProblemS{D},
  FEMInfo::FEMInfoS,
  Param::ParamInfoS) where D

  if "H" ∉ FEMInfo.probl_nl
    hₛ = x -> one(VectorValue(D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

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
  ::NTuple{3,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_neumann_datum(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(
    ∫(Param.α * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L(t) = (assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ * dg(t))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀))

  L

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(∫(Param.α * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g))*FEMSpace.dΩ,
    FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L₁(t) = assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ ⋅ dg(t))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀)

  L₁

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  L_stokes = assemble_lifting(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

  g = define_g_FEM(FEMSpace, Param)
  conv(u,∇u) = (∇u')⋅u
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )FEMSpace.dΩ
  L_convection = assemble_vector(c(g, FEMSpace.ϕᵥ), FEMSpace.V₀)

  (L_stokes + L_convection)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  L_stokes = assemble_lifting(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)
  L_stokes

end

function assemble_lifting_continuity(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(∫(FEMSpace.ψᵧ * (∇⋅g))*FEMSpace.dΩ,FEMSpace.Q₀)::Vector{Float}

end

function assemble_lifting_continuity(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  g = define_g_FEM(FEMSpace, Param)

  L₂(t) = assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(g(t))))*FEMSpace.dΩ,FEMSpace.Q₀)

  L₂

end

function assemble_lifting_continuity(
  ::NTuple{3,Int},
  FEMSpace::FEMProblem,
  FEMInfo::Info,
  Param::Info)

  assemble_lifting_continuity(get_NTuple(3, Int), FEMSpace, FEMInfo, Param)

end

function assemble_L²_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FEMProblem)

  assemble_matrix(∫(FEMSpace.ψᵧ*FEMSpace.ψₚ)*FEMSpace.dΩ,
    FEMSpace.Q, FEMSpace.Q₀)::SparseMatrixCSC{Float, Int}

end

function assemble_L²_norm_matrix(
  ::NTuple{3,Int},
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
  FEMSpace::FEMProblemS)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
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
    assemble_convection(NT,FEMSpace)
  elseif var == "D"
    assemble_swapped_convection(NT,FEMSpace)
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
