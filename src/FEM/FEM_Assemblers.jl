function assemble_form(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::ParamInfo)

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_form(FEMInfo.id, FEMSpace, FEMInfo, ParamForm)

end

function assemble_form(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  ParamForm::ParamFormInfo)

  assemble_form(FEMInfo.id, FEMSpace, FEMInfo, ParamForm)

end

function assemble_form(
  ::NTuple{1,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamFormInfoS)

  var = ParamForm.var

  function bilinear_form(u, v)
    if var == "Xu"
      ∫(∇(v) ⋅ ∇(u) + v * u)ParamForm.dΩ
    else var == "A"
      if isaffine(FEMInfo, var)
        ∫(∇(v) ⋅ ∇(u))ParamForm.dΩ
      else
        ∫(∇(v) ⋅ (ParamForm.fun * ∇(u)))ParamForm.dΩ
      end
    end
  end

  function linear_form(v)
    if var == "F" || "H"
      if isaffine(FEMInfo, var)
        ∫(v * (x->1.))ParamForm.dΩ
      else
        ∫(v * ParamForm.fun)ParamForm.dΩ
      end
    elseif var == "L"
      g = interpolate_dirichlet(ParamForm.fun, FEMSpace.V)
      Param_A = ParamInfo(FEMInfo, μ, "A")
      ∫(∇(v) ⋅ (Param_A.fun * ∇(g)))ParamForm.dΩ
    else
      error("Unrecognized variable")
    end
  end

  if var ∈ ("A", "Xu")
    bilinear_form
  else
    linear_form
  end

end

function assemble_form(
  ::NTuple{2,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  ParamForm::ParamFormInfoS)

  var = ParamForm.var

  function bilinear_form(u, v)
    if var == "Xu"
      ∫(∇(v) ⊙ ∇(u) + v ⋅ u)ParamForm.dΩ
    elseif var == "Xp"
      ∫(v * u)ParamForm.dΩ
    elseif var == "A"
      if isaffine(FEMInfo, var)
        ∫(∇(v) ⊙ ∇(u))ParamForm.dΩ
      else
        ∫(∇(v) ⊙ (ParamForm.fun * ∇(u)))ParamForm.dΩ
      end
    else var == "B"
      if isaffine(FEMInfo, var)
        ∫(v ⋅ ∇(u))ParamForm.dΩ
      else
        ∫(v ⋅ (ParamForm.fun * ∇(u)))ParamForm.dΩ
      end
    end
  end

  function linear_form(v)
    if var == "F" || "H"
      if isaffine(FEMInfo, var)
        ∫(v ⋅ (x->1.))ParamForm.dΩ
      else
        ∫(v ⋅ ParamForm.fun)ParamForm.dΩ
      end
    else
      g = interpolate_dirichlet(ParamForm.fun, FEMSpace.V)
      if var == "L"
        Param_AB = ParamInfo(FEMInfo, μ, "A")
      else var == "Lc"
        Param_AB = ParamInfo(FEMInfo, μ, "B")
      end
      ∫(∇(v) ⋅ (Param_AB.fun * ∇(g)))ParamForm.dΩ
    end
  end

  if var ∈ ("A", "B", "Xu", "Xp")
    bilinear_form
  else
    linear_form
  end

end

function assemble_form(
  ::NTuple{3,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  ParamForm::ParamFormInfoS)

  var = ParamForm.var

  function trilinear_form(u, v, z)
    if var == "C"
      ∫(v ⊙ (∇(u)'⋅z))ParamForm.dΩ
    else var == "D"
      ∫(v ⊙ (∇(z)'⋅u) )ParamForm.dΩ
    end
  end

  function bilinear_form(u, v)
    if var == "Xu"
      ∫(∇(v) ⊙ ∇(u) + v ⋅ u)ParamForm.dΩ
    elseif var == "Xp"
      ∫(v * u)ParamForm.dΩ
    elseif var == "A"
      if isaffine(FEMInfo, var)
        ∫(∇(v) ⊙ ∇(u))ParamForm.dΩ
      else
        ∫(∇(v) ⊙ (ParamForm.fun * ∇(u)))ParamForm.dΩ
      end
    else var == "B"
      if isaffine(FEMInfo, var)
        ∫(v ⋅ ∇(u))ParamForm.dΩ
      else
        ∫(v ⋅ (ParamForm.fun * ∇(u)))ParamForm.dΩ
      end
    end
  end

  function linear_form(v)
    if var == "F" || "H"
      if isaffine(FEMInfo, var)
        ∫(v ⋅ (x->1.))ParamForm.dΩ
      else
        ∫(v ⋅ ParamForm.fun)ParamForm.dΩ
      end
    else
      g = interpolate_dirichlet(ParamForm.fun, FEMSpace.V)
      if var == "L"
        Param_AB = ParamInfo(FEMInfo, μ, "A")
      else var == "Lc"
        Param_AB = ParamInfo(FEMInfo, μ, "B")
      end
      ∫(∇(v) ⋅ (Param_AB.fun * ∇(g)))ParamForm.dΩ
    end
  end

  if var ∈ ("C", "D")
    trilinear_form
  elseif var ∈ ("A", "B", "Xu", "Xp")
    bilinear_form
  else
    linear_form
  end

end

#= function assemble_mass(
  ::NTuple{1,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_mass(t)
    if "M" ∉ FEMInfo.affine_structures
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
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_mass(t)
    if "M" ∉ FEMInfo.affine_structures
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
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  assemble_mass(NTuple(2, Int), FEMSpace, FEMInfo, Param)

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  if "A" ∉ FEMInfo.affine_structures
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{1,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_stiffness(t)
    if "A" ∉ FEMInfo.affine_structures
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
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  if "A" ∉ FEMInfo.affine_structures
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  else
    assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀)
  end

end

function assemble_stiffness(
  ::NTuple{2,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_stiffness(t)
    if "A" ∉ FEMInfo.affine_structures
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
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::FOMInfo)

  assemble_stiffness(NTuple(2, Int), FEMSpace, FEMInfo, Param)

end

function assemble_B(
  ::NTuple{2,Int},
  FEMSpace::FOMS,
  ::FOMInfo,
  ::FOMInfo)

  if "B" ∉ FEMInfo.affine_structures
    assemble_matrix(∫(FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.Q₀)
  else
    assemble_matrix(∫(Param.b * FEMSpace.ψᵧ*(∇⋅(FEMSpace.ϕᵤ)))*FEMSpace.dΩ,
      FEMSpace.V, FEMSpace.Q₀)
  end

end

function assemble_B(
  ::NTuple{2,Int},
  FEMSpace::FOMST,
  ::FOMInfo,
  ::FOMInfo)

  function unsteady_primal_form(t)
    if "B" ∉ FEMInfo.affine_structures
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
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::FOMInfo)

  assemble_B(NTuple(2, Int), FEMSpace, FEMInfo, Param)

end

function assemble_convection(
  ::NTuple{3,Int},
  FEMSpace::FOMS)

  C(u) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(FEMSpace.ϕᵤ)'⋅u) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  C

end

function assemble_convection(
  ::NTuple{3,Int},
  FEMSpace::FOMST)

  C(u,t) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(FEMSpace.ϕᵤ(t))'⋅u(t)) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  C

end

function assemble_swapped_convection(
  ::NTuple{3,Int},
  FEMSpace::FOMS)

  D(u) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(u)'⋅FEMSpace.ϕᵤ) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

  D

end

function assemble_swapped_convection(
  ::NTuple{3,Int},
  FEMSpace::FOMST)

  D(u,t) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(u(t))'⋅FEMSpace.ϕᵤ(t)) )*FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  D

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  if "F" ∉ FEMInfo.affine_structures
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{1,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_forcing(t)
    if "F" ∉ FEMInfo.affine_structures
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
    else
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.f(t))*FEMSpace.dΩ, FEMSpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_forcing(
  ::NTuple{2,Int},
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS,
  Param::ParamInfoS) where D

  if "F" ∉ FEMInfo.affine_structures
    fₛ = x -> one(VectorValue(D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*FEMSpace.dΩ, FEMSpace.V₀)
  end

end

function assemble_forcing(
  ::NTuple{2,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_forcing(t)
    if "F" ∉ FEMInfo.affine_structures
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.fₛ)*FEMSpace.dΩ, FEMSpace.V₀)
    else
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f(t))*FEMSpace.dΩ, FEMSpace.V₀)
    end
  end

  unsteady_forcing

end

function assemble_forcing(
  ::NTuple{3,Int},
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::FOMInfo)

  assemble_forcing(NTuple(2, Int), FEMSpace, FEMInfo, Param)

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  if "H" ∉ FEMInfo.affine_structures
    assemble_vector(∫(FEMSpace.ϕᵥ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{1,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_neumann_datum(t)

    if "H" ∉ FEMInfo.affine_structures
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    else
      assemble_vector(∫(FEMSpace.ϕᵥ*Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    end

  end

  unsteady_neumann_datum

end

function assemble_neumann_datum(
  ::NTuple{2,Int},
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS,
  Param::ParamInfoS) where D

  if "H" ∉ FEMInfo.affine_structures
    hₛ = x -> one(VectorValue(D, Float))
    assemble_vector(∫(FEMSpace.ϕᵥ⋅hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  else
    assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
  end

end

function assemble_neumann_datum(
  ::NTuple{2,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function unsteady_neumann_datum(t)
    if "H" ∉ FEMInfo.affine_structures
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.hₛ)*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    else
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h(t))*FEMSpace.dΓn, FEMSpace.V₀)::Vector{Float}
    end
  end

  unsteady_neumann_datum

end

function assemble_neumann_datum(
  ::NTuple{3,Int},
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::FOMInfo)

  assemble_neumann_datum(NTuple(2, Int), FEMSpace, FEMInfo, Param)

end

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(
    ∫(Param.α * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g))*FEMSpace.dΩ,FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{1,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L(t) = (assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ * dg(t))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀))

  L

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(∫(Param.α * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g))*FEMSpace.dΩ,
    FEMSpace.V₀)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{2,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  g = define_g_FEM(FEMSpace, Param)
  dg = define_dg_FEM(FEMSpace, Param)

  L₁(t) = assemble_vector(∫(Param.m(t) * FEMSpace.ϕᵥ ⋅ dg(t))*FEMSpace.dΩ,FEMSpace.V₀) +
    assemble_vector(∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g(t)))*FEMSpace.dΩ,FEMSpace.V₀)

  L₁

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  L_stokes = assemble_lifting(NTuple(2, Int), FEMSpace, FEMInfo, Param)

  g = define_g_FEM(FEMSpace, Param)
  conv(u,∇u) = (∇u')⋅u
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )FEMSpace.dΩ
  L_convection = assemble_vector(c(g, FEMSpace.ϕᵥ), FEMSpace.V₀)

  (L_stokes + L_convection)::Vector{Float}

end

function assemble_lifting(
  ::NTuple{3,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  L_stokes = assemble_lifting(NTuple(2, Int), FEMSpace, FEMInfo, Param)
  L_stokes

end

function assemble_lifting_continuity(
  ::NTuple{2,Int},
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  g = define_g_FEM(FEMSpace, Param)

  assemble_vector(∫(FEMSpace.ψᵧ * (∇⋅g))*FEMSpace.dΩ,FEMSpace.Q₀)::Vector{Float}

end

function assemble_lifting_continuity(
  ::NTuple{2,Int},
  FEMSpace::FOMST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  g = define_g_FEM(FEMSpace, Param)

  L₂(t) = assemble_vector(∫(FEMSpace.ψᵧ*(∇⋅(g(t))))*FEMSpace.dΩ,FEMSpace.Q₀)

  L₂

end

function assemble_lifting_continuity(
  ::NTuple{3,Int},
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::FOMInfo)

  assemble_lifting_continuity(NTuple(2, Int), FEMSpace, FEMInfo, Param)

end

function assemble_L²_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FOM)

  assemble_matrix(∫(FEMSpace.ψᵧ*FEMSpace.ψₚ)*FEMSpace.dΩ,
    FEMSpace.Q, FEMSpace.Q₀)::SparseMatrixCSC{Float, Int}

end

function assemble_L²_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FOM)

  assemble_L²_norm_matrix(NTuple(2, Int), FEMSpace)

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::FOMS)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ)*FEMSpace.dΩ,
    FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{1,Int},
  FEMSpace::FOMST)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ*FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FOMS)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ))*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀) +
  assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ)*FEMSpace.dΩ,
  FEMSpace.V, FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{2,Int},
  FEMSpace::FOMST)

  (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙∇(FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀) +
    assemble_matrix(∫(FEMSpace.ϕᵥ⋅FEMSpace.ϕᵤ(0.0))*FEMSpace.dΩ,
    FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{Float, Int}

end

function assemble_H¹_norm_matrix(
  ::NTuple{3,Int},
  FEMSpace::FOM)

  assemble_H¹_norm_matrix(NTuple(2, Int), FEMSpace)

end =#

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  fun::Function,
  var::String)

  Param = ParamInfo(FEMSpace, fun, var)
  assemble_FEM_structure(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::ParamInfo)

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_structure(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  μ::Vector{T},
  var::String) where T

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_structure(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  μvec::Vector{Vector{T}},
  var::String) where T

  MV(μ) = assemble_FEM_structure(FEMSpace, FEMInfo, μ, var)
  Broadcasting(MV)(μvec)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  Param::Vector{ParamInfo})

  FEM_structure(P) = assemble_FEM_structure(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_structure)(Param)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  ParamForm::ParamFormInfo)

  var = ParamForm.var

  form = assemble_form(FEMSpace, FEMInfo, ParamForm)

  if var ∈ get_FEM_vectors(FEMInfo)
    assemble_vector(form, FEMSpace_vectors(FEMSpace, var))
  else
    assemble_vector(form, FEMSpace_matrices(FEMSpace, var)...)
  end

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  FEMInfo::FOMInfo,
  ParamForm::Vector{ParamFormInfo})

  FEM_structure(P) = assemble_FEM_structure(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_structure)(ParamForm)

end
