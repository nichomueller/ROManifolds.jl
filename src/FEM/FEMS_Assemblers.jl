function assemble_form(
  FEMSpace::FOMS{1,D},
  FEMInfo::FOMInfoS{1},
  ParamForm::ParamFormInfoS) where D

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
    if var ∈ ("F", "H")
      if isaffine(FEMInfo, var)
        ∫(v * get_g₁(FEMInfo))ParamForm.dΩ
      else
        ∫(v * ParamForm.fun)ParamForm.dΩ
      end
    elseif var == "L"
      g = interpolate_dirichlet(ParamForm.fun, FEMSpace.V[1])
      Param_A = ParamInfo(FEMInfo, ParamForm.μ, "A")
      ∫(∇(v) ⋅ (Param_A.fun * ∇(g)))ParamForm.dΩ
    else
      error("Unrecognized variable")
    end
  end

  var ∈ ("A", "Xu") ? bilinear_form : linear_form

end

function assemble_form(
  FEMSpace::FOMS{2,D},
  FEMInfo::FOMInfoS{2},
  ParamForm::ParamFormInfoS) where D

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
        ∫(v * (∇⋅(u)))ParamForm.dΩ
      else
        ∫(ParamForm.fun * v * (∇⋅(u)))ParamForm.dΩ
      end
    end
  end

  function linear_form(v)
    if var ∈ ("F", "H")
      if isaffine(FEMInfo, var)
        ∫(v ⋅ get_g₁(FEMInfo))ParamForm.dΩ
      else
        ∫(v ⋅ ParamForm.fun)ParamForm.dΩ
      end
    else
      g = interpolate_dirichlet(ParamForm.fun, FEMSpace.V[1])
      if var == "L"
        Param_A = ParamInfo(FEMInfo, ParamForm.μ, "A")
        ∫(Param_A.fun * ∇(v) ⊙ ∇(g))ParamForm.dΩ
      else var == "Lc"
        Param_B = ParamInfo(FEMInfo, ParamForm.μ, "B")
        ∫(Param_B.fun * v ⋅ (∇⋅(g)))ParamForm.dΩ
      end
    end
  end

  var ∈ ("A", "B", "Xu", "Xp") ? bilinear_form : linear_form

end

function assemble_form(
  FEMSpace::FOMS{3,D},
  FEMInfo::FOMInfoS{3},
  ParamForm::ParamFormInfoS) where D

  var = ParamForm.var

  function trilinear_form(u, v, z)
    if var == "C"
      ∫(v ⊙ (∇(u)'⋅z))ParamForm.dΩ
    else var == "D"
      ∫(v ⊙ (∇(z)'⋅u))ParamForm.dΩ
    end
  end
  trilinear_form(z) = (u, v) -> trilinear_form(u, v, z)

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
        ∫(v * (∇⋅(u)))ParamForm.dΩ
      else
        ∫(ParamForm.fun * v * (∇⋅(u)))ParamForm.dΩ
      end
    end
  end

  function linear_form(v)
    if var ∈ ("F", "H")
      if isaffine(FEMInfo, var)
        ∫(v ⋅ get_g₁(FEMInfo))ParamForm.dΩ
      else
        ∫(v ⋅ ParamForm.fun)ParamForm.dΩ
      end
    else
      g = interpolate_dirichlet(ParamForm.fun, FEMSpace.V[1])
      if var == "L"
        Param_A = ParamInfo(FEMInfo, ParamForm.μ, "A")
        conv(u,∇u) = (∇u')⋅u
        c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )ParamForm.dΩ
        ∫(Param_A.fun * ∇(v) ⊙ ∇(g))ParamForm.dΩ + c(g,v)
      else var == "Lc"
        Param_B = ParamInfo(FEMInfo, ParamForm.μ, "B")
        ∫(Param_B.fun * v ⋅ (∇⋅(g)))ParamForm.dΩ
      end
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

function assemble_form(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_form(FEMSpace, FEMInfo, ParamForm)

end

#####################################LINEAR#####################################

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  ParamForm::ParamFormInfo) where {ID,D}


  var = ParamForm.var
  form = assemble_form(FEMSpace, FEMInfo, ParamForm)
  Mat = assemble_matrix(form, get_FEMSpace_matrix(FEMSpace, var)...)
  Mat::SparseMatrixCSC{Float,Int}

end

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  ParamForm::Vector{<:ParamFormInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_matrix(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_matrix)(ParamForm)

end

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::Vector{<:ParamInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_matrix(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_matrix)(Param)

end

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T},
  var::String) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μvec::Vector{Vector{T}},
  var::String) where {ID,D,T}

  Mat(μ) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Mat)(μvec)

end

function assemble_FEM_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T},
  var::Vector{String}) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Param)

end

###################################NONLINEAR####################################

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  ParamForm::ParamFormInfo) where {ID,D}

    var = ParamForm.var
    form(z) = assemble_form(FEMSpace, FEMInfo, ParamForm)(z)
    Mat(z) = assemble_matrix(form(z), get_FEMSpace_matrix(FEMSpace, var)...)
    Mat

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  ParamForm::Vector{<:ParamFormInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_matrix)(ParamForm)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::Vector{<:ParamInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_matrix)(Param)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T},
  var::String) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μvec::Vector{Vector{T}},
  var::String) where {ID,D,T}

  Mat(μ) = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Mat)(μvec)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T},
  var::Vector{String}) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param)

end

#####################################LINEAR#####################################

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  ParamForm::ParamFormInfo) where {ID,D}

  var = ParamForm.var
  form = assemble_form(FEMSpace, FEMInfo, ParamForm)
  if var ∈ ("L", "Lc")
    Vec = -assemble_vector(form, get_FEMSpace_vector(FEMSpace, var))
  else
    Vec = assemble_vector(form, get_FEMSpace_vector(FEMSpace, var))
  end

  Vec::Vector{Float}

end

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  ParamForm::Vector{<:ParamFormInfo}) where {ID,D}

  FEM_vector(P) = assemble_FEM_vector(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_vector)(ParamForm)

end

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  Param::Vector{<:ParamInfo}) where {ID,D}

  FEM_vector(P) = assemble_FEM_vector(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_vector)(Param)

end

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T},
  var::String) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_vector(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μvec::Vector{Vector{T}},
  var::String) where {ID,D,T}

  Vec(μ) = assemble_FEM_vector(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Vec)(μvec)

end

function assemble_FEM_vector(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T},
  var::Vector{String}) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_vector(FEMSpace, FEMInfo, Param)

end

#####################################EXTRAS#####################################

function assemble_affine_FEM_matrices(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_affine_matrices(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Params)

end

function assemble_all_FEM_matrices(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_FEM_matrices(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Params)

end

function assemble_affine_FEM_vectors(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_affine_vectors(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_vector(FEMSpace, FEMInfo, Params)

end

function assemble_all_FEM_vectors(
  FEMSpace::FOMS{ID,D},
  FEMInfo::FOMInfoS{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_FEM_vectors(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_vector(FEMSpace, FEMInfo, Params)

end
