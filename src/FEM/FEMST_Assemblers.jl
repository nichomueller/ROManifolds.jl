function assemble_form(
  FEMSpace::FOMST{1,D},
  FEMInfo::FOMInfoST{1},
  ParamForm::ParamFormInfoST) where D

  var = ParamForm.var

  function bilinear_form(u, v, t)
    if var == "Xu"
      ∫(∇(v) ⋅ ∇(u) + v * u)ParamForm.dΩ
    elseif var == "A"
      if isaffine(FEMInfo, var)
        ∫(∇(v) ⋅ ∇(u))ParamForm.dΩ
      else
        ∫(∇(v) ⋅ (ParamForm.fun(t) * ∇(u)))ParamForm.dΩ
      end
    else var == "M"
      if isaffine(FEMInfo, var)
        ∫(v * u)ParamForm.dΩ
      else
        ∫(ParamForm.fun(t) * v * u)ParamForm.dΩ
      end
    end
  end
  bilinear_form(t) = (u, v) -> bilinear_form(u, v, t)

  function linear_form(v, t)
    if var ∈ ("F", "H")
      if isaffine(FEMInfo, var)
        ∫(v * get_g₁(FEMInfo)(t))ParamForm.dΩ
      else
        ∫(v * ParamForm.fun(t))ParamForm.dΩ
      end
    elseif var == "L"
      g(t) = interpolate_dirichlet(ParamForm.fun(t), FEMSpace.V[1](t))
      Param_A = ParamInfo(FEMInfo, ParamForm.μ, "A")
      ∫(∇(v) ⋅ (Param_A.fun(t) * ∇(g(t))))ParamForm.dΩ
    else
      error("Unrecognized variable")
    end
  end
  linear_form(t) = v -> linear_form(v, t)

  var ∈ ("A", "M", "Xu") ? bilinear_form : linear_form

end

function assemble_form(
  FEMSpace::FOMST{2,D},
  FEMInfo::FOMInfoST{2},
  ParamForm::ParamFormInfoST) where D

  var = ParamForm.var

  function bilinear_form(u, v, t)
    if var == "Xu"
      ∫(∇(v) ⊙ ∇(u) + v ⋅ u)ParamForm.dΩ
    elseif var == "Xp"
      ∫(v * u)ParamForm.dΩ
    elseif var == "A"
      if isaffine(FEMInfo, var)
        ∫(∇(v) ⊙ ∇(u))ParamForm.dΩ
      else
        ∫(∇(v) ⊙ (ParamForm.fun(t) * ∇(u)))ParamForm.dΩ
      end
    elseif var == "B"
      if isaffine(FEMInfo, var)
        ∫(v * (∇⋅(u)))ParamForm.dΩ
      else
        ∫(ParamForm.fun(t) * v * (∇⋅(u)))ParamForm.dΩ
      end
    else var == "M"
      if isaffine(FEMInfo, var)
        ∫(v ⋅ u)ParamForm.dΩ
      else
        ∫(ParamForm.fun(t) * v ⋅ u)ParamForm.dΩ
      end
    end
  end
  bilinear_form(t) = (u, v) -> bilinear_form(u, v, t)

  function linear_form(v, t)
    if var ∈ ("F", "H")
      if isaffine(FEMInfo, var)
        ∫(v ⋅ get_g₁(FEMInfo)(t))ParamForm.dΩ
      else
        ∫(v ⋅ ParamForm.fun(t))ParamForm.dΩ
      end
    else
      g(t) = interpolate_dirichlet(ParamForm.fun(t), FEMSpace.V[1](t))
      if var == "L"
        Param_A = ParamInfo(FEMInfo, ParamForm.μ, "A")
        ∫(Param_A.fun(t) * ∇(v) ⊙ ∇(g(t)))ParamForm.dΩ
      else var == "Lc"
        Param_B = ParamInfo(FEMInfo, ParamForm.μ, "B")
        ∫(Param_B.fun(t) * v ⋅ (∇⋅(g(t))))ParamForm.dΩ
      end
    end
  end
  linear_form(t) = v -> linear_form(v, t)

  var ∈ ("A", "B", "M", "Xu", "Xp") ? bilinear_form : linear_form

end

function assemble_form(
  FEMSpace::FOMST{3,D},
  FEMInfo::FOMInfoST{3},
  ParamForm::ParamFormInfoST) where D

  var = ParamForm.var

  function trilinear_form(u, v, z, t)
    if var == "C"
      ∫(v ⊙ (∇(u)'⋅z(t)))ParamForm.dΩ
    else var == "D"
      ∫(v ⊙ (∇(z(t))'⋅u) )ParamForm.dΩ
    end
  end
  trilinear_form(z, t) = (u, v) -> trilinear_form(u, v, z, t)
  trilinear_form(t) = z -> trilinear_form(z, t)

  function bilinear_form(u, v, t)
    if var == "Xu"
      ∫(∇(v) ⊙ ∇(u) + v ⋅ u)ParamForm.dΩ
    elseif var == "Xp"
      ∫(v * u)ParamForm.dΩ
    elseif var == "A"
      if isaffine(FEMInfo, var)
        ∫(∇(v) ⊙ ∇(u))ParamForm.dΩ
      else
        ∫(∇(v) ⊙ (ParamForm.fun(t) * ∇(u)))ParamForm.dΩ
      end
    elseif var == "B"
      if isaffine(FEMInfo, var)
        ∫(v * (∇⋅(u)))ParamForm.dΩ
      else
        ∫(ParamForm.fun(t) * v * (∇⋅(u)))ParamForm.dΩ
      end
    else var == "M"
      if isaffine(FEMInfo, var)
        ∫(v ⋅ u)ParamForm.dΩ
      else
        ∫(ParamForm.fun(t) * v ⋅ u)ParamForm.dΩ
      end
    end
  end
  bilinear_form(t) = (u, v) -> bilinear_form(u, v, t)

  function linear_form(v, t)
    if var ∈ ("F", "H")
      if isaffine(FEMInfo, var)
        ∫(v ⋅ get_g₁(FEMInfo)(t))ParamForm.dΩ
      else
        ∫(v ⋅ ParamForm.fun(t))ParamForm.dΩ
      end
    else
      g(t) = interpolate_dirichlet(ParamForm.fun(t), FEMSpace.V[1](t))
      if var == "L"
        Param_A = ParamInfo(FEMInfo, ParamForm.μ, "A")
        ∫(Param_A.fun(t) * ∇(v) ⊙ ∇(g(t)))ParamForm.dΩ
      else var == "Lc"
        Param_B = ParamInfo(FEMInfo, ParamForm.μ, "B")
        ∫(Param_B.fun(t) * v ⋅ (∇⋅(g(t))))ParamForm.dΩ
      end
    end
  end
  linear_form(t) = v -> linear_form(v, t)

  if var ∈ ("C", "D")
    trilinear_form
  elseif var ∈ ("A", "B", "M", "Xu", "Xp")
    bilinear_form
  else
    linear_form
  end

end

function assemble_form(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_form(FEMSpace, FEMInfo, ParamForm)

end

#####################################LINEAR#####################################

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo) where {ID,D}

  var = ParamForm.var
  form = assemble_form(FEMSpace, FEMInfo, ParamForm)
  V, V₀ = get_FEMSpace_matrix(FEMSpace, var)
  t -> assemble_matrix(form(t), V(t), V₀)::SparseMatrixCSC{Float,Int}

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_matrix(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_matrix)(ParamForm)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_matrix(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_matrix)(Param)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String) where {ID,D,T}

  Mat(μ) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Mat)(μvec)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String}) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Param)

end

################################################################################

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo,
  tθ::Real) where {ID,D}

  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)(tθ)::SparseMatrixCSC{Float,Int}

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo},
  tθ::Real) where {ID,D}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo,
  tθ::Real) where {ID,D}

  assemble_FEM_matrix(FEMSpace, FEMInfo, Param)(tθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo},
  tθ::Real) where {ID,D}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, Param)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String,
  tθ::Real) where {ID,D,T}

  assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var)(tθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String,
  tθ::Real) where {ID,D,T}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, μvec, var)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String},
  tθ::Real) where {ID,D,T}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo,
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm, tθ))(timesθ)::Vector{SparseMatrixCSC{Float,Int}}

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo},
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm, tθ))(timesθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo,
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, Param, tθ))(timesθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo},
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, Param, tθ))(timesθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String,
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var, tθ))(timesθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String,
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, μvec, var, tθ))(timesθ)

end

function assemble_FEM_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var, tθ))(timesθ)

end

###################################NONLINEAR####################################

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo) where {ID,D}

  var = ParamForm.var
  form = assemble_form(FEMSpace, FEMInfo, ParamForm)
  V, V₀ = get_FEMSpace_matrix(FEMSpace, var)
  t -> assemble_matrix(form(t), V(t), V₀)::SparseMatrixCSC{Float,Int}

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_matrix)(ParamForm)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo}) where {ID,D}

  FEM_matrix(P) = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_matrix)(Param)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String) where {ID,D,T}

  Mat(μ) = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Mat)(μvec)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String}) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param)

end

################################################################################

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo,
  tθ::Real) where {ID,D}

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm)(tθ)::SparseMatrixCSC{Float,Int}

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo},
  tθ::Real) where {ID,D}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo,
  tθ::Real) where {ID,D}

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param)(tθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo},
  tθ::Real) where {ID,D}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, Param)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String,
  tθ::Real) where {ID,D,T}

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, var)(tθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String,
  tθ::Real) where {ID,D,T}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, μvec, var)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String},
  tθ::Real) where {ID,D,T}

  Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo,
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm, tθ))(timesθ)::Vector{SparseMatrixCSC{Float,Int}}

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo},
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm, tθ))(timesθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo,
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param, tθ))(timesθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo},
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, Param, tθ))(timesθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String,
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, var, tθ))(timesθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String,
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μvec, var, tθ))(timesθ)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, var, tθ))(timesθ)

end

#####################################LINEAR#####################################

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo) where {ID,D}

  var = ParamForm.var
  form = assemble_form(FEMSpace, FEMInfo, ParamForm)

  function Vec(t)
    if var ∈ ("L", "Lc")
      -assemble_vector(form(t), get_FEMSpace_vector(FEMSpace, var))::Vector{Float}
    else
      assemble_vector(form(t), get_FEMSpace_vector(FEMSpace, var))::Vector{Float}
    end
  end

  Vec

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo}) where {ID,D}

  FEM_vector(P) = assemble_FEM_vector(FEMSpace, FEMInfo, P)
  Broadcasting(FEM_vector)(ParamForm)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo) where {ID,D}

  ParamForm = ParamFormInfo(FEMSpace, Param)
  assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo}) where {ID,D}

  FEM_vector(P) = assemble_FEM_vector(FEMSpace, FEMInfo,
    ParamFormInfo(FEMSpace, P))
  Broadcasting(FEM_vector)(Param)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_vector(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String) where {ID,D,T}

  Vec(μ) = assemble_FEM_vector(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Vec)(μvec)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String}) where {ID,D,T}

  Param = ParamInfo(FEMInfo, μ, var)
  assemble_FEM_vector(FEMSpace, FEMInfo, Param)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo,
  tθ::Real) where {ID,D}

  Vector{Float}(assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)(tθ))::Vector{Float}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo},
  tθ::Real) where {ID,D}

  Vecs = assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)
  Broadcasting(Vec -> Gridap.evaluate(Vec, tθ))(Vecs)::Vector{Vector{Float}}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo,
  tθ::Real) where {ID,D}

  Vector{Float}(assemble_FEM_vector(FEMSpace, FEMInfo, Param)(tθ))::Vector{Float}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo},
  tθ::Real) where {ID,D}

  Vecs = assemble_FEM_vector(FEMSpace, FEMInfo, Param)
  Broadcasting(Vec -> Gridap.evaluate(Vec, tθ))(Vecs)::Vector{Vector{Float}}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String,
  tθ::Real) where {ID,D,T}

  Vector{Float}(assemble_FEM_vector(FEMSpace, FEMInfo, μ, var)(tθ))::Vector{Float}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String,
  tθ::Real) where {ID,D,T}

  Vecs = assemble_FEM_vector(FEMSpace, FEMInfo, μvec, var)
  Broadcasting(Vec -> Gridap.evaluate(Vec, tθ))(Vecs)::Vector{Vector{Float}}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String},
  tθ::Real) where {ID,D,T}

  Vecs = assemble_FEM_vector(FEMSpace, FEMInfo, μ, var)
  Broadcasting(Vec -> Gridap.evaluate(Vec, tθ))(Vecs)::Vector{Vector{Float}}

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::ParamFormInfo,
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm, tθ))(timesθ)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  ParamForm::Vector{<:ParamFormInfo},
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm, tθ))(timesθ)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfo,
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, Param, tθ))(timesθ)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  Param::Vector{<:ParamInfo},
  timesθ::Vector{<:Real}) where {ID,D}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, Param, tθ))(timesθ)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::String,
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, μ, var, tθ))(timesθ)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μvec::Vector{Vector{T}},
  var::String,
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, μvec, var, tθ))(timesθ)

end

function assemble_FEM_vector(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  var::Vector{String},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_FEM_vector(FEMSpace, FEMInfo, μ, var, tθ))(timesθ)

end

#####################################EXTRAS#####################################

function assemble_affine_FEM_matrices(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_affine_matrices(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Params)

end

function assemble_affine_FEM_matrices(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  tθ::Real) where {ID,D,T}

  Mats = assemble_affine_FEM_matrices(FEMSpace, FEMInfo, μ)
  if isempty(Mats)
    Vector{SparseMatrixCSC{Float,Int}}[]
  else
    Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)::Vector{SparseMatrixCSC{Float,Int}}
  end

end

function assemble_affine_FEM_matrices(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_affine_FEM_matrices(FEMSpace, FEMInfo, μ, tθ))(timesθ)

end

function assemble_all_FEM_matrices(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_FEM_matrices(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_matrix(FEMSpace, FEMInfo, Params)

end

function assemble_all_FEM_matrices(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  tθ::Real) where {ID,D,T}

  Mats = assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ)
  if isempty(Mats)
    Vector{SparseMatrixCSC{Float,Int}}[]
  else
    Broadcasting(Mat -> Gridap.evaluate(Mat, tθ))(Mats)::Vector{SparseMatrixCSC{Float,Int}}
  end

end

function assemble_all_FEM_matrices(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ, tθ))(timesθ)

end

function assemble_affine_FEM_vectors(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_affine_vectors(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_vector(FEMSpace, FEMInfo, Params)

end

function assemble_affine_FEM_vectors(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  tθ::Real) where {ID,D,T}

  Vecs = assemble_affine_FEM_vectors(FEMSpace, FEMInfo, μ)
  if isempty(Vecs)
    Vector{Vector{Float64}}[]
  else
    Broadcasting(Vec -> Gridap.evaluate(Vec, tθ))(Vecs)::Vector{Vector{Float}}
  end

end

function assemble_affine_FEM_vectors(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_affine_FEM_vectors(FEMSpace, FEMInfo, μ, tθ))(timesθ)

end

function assemble_all_FEM_vectors(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T}) where {ID,D,T}

  operators = get_FEM_vectors(FEMInfo)
  Params = ParamInfo(FEMInfo, μ, operators)
  assemble_FEM_vector(FEMSpace, FEMInfo, Params)

end

function assemble_all_FEM_vectors(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  tθ::Real) where {ID,D,T}

  Vecs = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ)
  if isempty(Vecs)
    Vector{Vector{Float64}}[]
  else
    Broadcasting(Vec -> Gridap.evaluate(Vec, tθ))(Vecs)::Vector{Vector{Float}}
  end

end

function assemble_all_FEM_vectors(
  FEMSpace::FOMST{ID,D},
  FEMInfo::FOMInfoST{ID},
  μ::Vector{T},
  timesθ::Vector{<:Real}) where {ID,D,T}

  Broadcasting(tθ ->
    assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ, tθ))(timesθ)

end
