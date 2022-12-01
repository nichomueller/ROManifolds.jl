function get_parameter()

end

function mdeim_online(
  Mat_nonaffine::AbstractArray{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  θvec = Matᵢ \ reshape(Mat_nonaffine, :, Nₜ)[idx,:]
  blocks(Matrix{T}(θvec'))
end

function mdeim_online(
  Mats_nonaffine::Vector{<:AbstractArray{T}},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  Mat_nonaffine = Matrix(Mats_nonaffine)
  MDEIM_online(Mat_nonaffine, Matᵢ, idx, Nₜ)
end

function mdeim_online(
  Fun_nonaffine::Function,
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  θmat(u) = Matᵢ \ reshape(Fun_nonaffine(u), :, Nₜ)[idx,:]
  θblock(u) = blocks(Matrix{T}(θmat(u)'))

  θblock
end
















function assemble_hyperred_matrix(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  Ω_hyp = view(FEMSpace.Ω, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_hyperred_vector(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  triang = Gridap.FESpaces.get_triangulation(FEMSpace, Param.var)
  Ω_hyp = view(triang, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)

end

function assemble_hyperred_fun_mat(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  Ω_hyp = view(FEMSpace.Ω, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_hyperred_fun_vec(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  triang = Gridap.FESpaces.get_triangulation(FEMSpace, Param.var)
  Ω_hyp = view(triang, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_nonlinear_vector(FEMSpace, FEMInfo, ParamForm)

end

function assemble_hyperred_matrix(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  Ω_hyp = view(FEMSpace.Ω, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm, timesθ)

end

function assemble_hyperred_vector(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  triang = Gridap.FESpaces.get_triangulation(FEMSpace, Param.var)
  Ω_hyp = view(triang, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm, timesθ)

end

function assemble_hyperred_fun_mat(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  Ω_hyp = view(FEMSpace.Ω, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm, timesθ)

end


function assemble_hyperred_fun_vec(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  triang = Gridap.FESpaces.get_triangulation(FEMSpace, Param.var)
  Ω_hyp = view(triang, el)
  dΩ_hyp = Measure(Ω_hyp, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_hyp)

  assemble_FEM_nonlinear_vector(FEMSpace, FEMInfo, ParamForm, timesθ)

end

function θ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, T))[1]]]
  else
    Mat_μ_hyp =
      assemble_hyperred_matrix(FEMSpace, FEMInfo, Param, MDEIM.el)
    θ = MDEIM_online(Mat_μ_hyp, MDEIM.Matᵢ, MDEIM.idx)
  end

  θ::Vector{Vector{T}}

end

function θ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::VMDEIM{T}) where {ID,D,T}

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, T))[1]]]
  else
    Vec_μ_hyp =
      assemble_hyperred_vector(FEMSpace, FEMInfo, Param, MDEIM.el)
    θ = MDEIM_online(Vec_μ_hyp, MDEIM.Matᵢ, MDEIM.idx)
  end

  θ::Vector{Vector{T}}

end

function θ_function(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  @assert isnonlinear(RBInfo, Param.var) "This method is only for nonlinear variables"

  Fun_μ_hyp =
    assemble_hyperred_fun_mat(FEMSpace, FEMInfo, Param, MDEIM.el)
  MDEIM_online(Fun_μ_hyp, MDEIM.Matᵢ, MDEIM.idx)

end

function θ_function(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::VMDEIM{T}) where {ID,D,T}

  @assert isnonlinear(RBInfo, Param.var) "This method is only for nonlinear variables"

  Fun_μ_hyp =
    assemble_hyperred_fun_vec(FEMSpace, FEMInfo, Param, MDEIM.el)
  MDEIM_online(Fun_μ_hyp, MDEIM.Matᵢ, MDEIM.idx)

end

function interpolate_θ(
  Mat_μ_hyp::AbstractArray{T},
  MDEIM::MVMDEIM{T},
  timesθ::Vector{T}) where T

  idx, time_idx = MDEIM.idx, MDEIM.time_idx
  red_timesθ = timesθ[time_idx]
  discarded_time_idx = setdiff(eachindex(timesθ), time_idx)
  θ = zeros(T, length(idx), length(timesθ))

  red_θ = (MDEIM.Matᵢ \
    Matrix{T}(reshape(Mat_μ_hyp, :, length(red_timesθ))[idx, :]))
  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ, 1, :), red_θ')
  θ[:, time_idx] = red_θ
  for iₜ = discarded_time_idx
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  blocks(Matrix{T}(θ'))

end

function interpolate_θ(
  Mats_μ_hyp::Vector{<:AbstractArray{T}},
  MDEIM::MVMDEIM{T},
  timesθ::Vector{T}) where T

  Mat_μ_hyp = Matrix(Mats_μ_hyp)
  interpolate_θ(Mat_μ_hyp, MDEIM, timesθ)

end

function θ(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  Param::ParamInfoST,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.funₜ(tθ) for tθ in timesθ]]
  else
    if RBInfo.st_mdeim
      red_timesθ = timesθ[MDEIM.time_idx]
      Mats_μ_hyp = assemble_hyperred_matrix(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ)
      θ = interpolate_θ(Mats_μ_hyp, MDEIM, timesθ)
    else
      Mats_μ_hyp = assemble_hyperred_matrix(
        FEMSpace, FEMInfo, Param, MDEIM.el, timesθ)
      θ = MDEIM_online(Mats_μ_hyp, MDEIM.Matᵢ, MDEIM.idx, length(timesθ))
    end
  end

  θ::Vector{Vector{T}}

end

function θ(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  Param::ParamInfoST,
  MDEIM::VMDEIM{T}) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.funₜ(tθ) for tθ in timesθ]]
  else
    if RBInfo.st_mdeim
      red_timesθ = timesθ[MDEIM.time_idx]
      Vecs_μ_hyp = assemble_hyperred_vector(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ)
      θ = interpolate_θ(Vecs_μ_hyp, MDEIM, timesθ)
    else
      Vecs_μ_hyp = assemble_hyperred_vector(
        FEMSpace, FEMInfo, Param, MDEIM.el, timesθ)
      θ = MDEIM_online(Vecs_μ_hyp, MDEIM.Matᵢ, MDEIM.idx, length(timesθ))
    end
  end

  θ::Vector{Vector{T}}

end

function θ_function(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  Param::ParamInfoST,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  @assert isnonlinear(RBInfo, Param.var) "This method is only for nonlinear variables"

  if RBInfo.st_mdeim
    red_timesθ = timesθ[MDEIM.time_idx]
    Fun_μ_hyp = assemble_hyperred_fun_mat(
      FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ)
    interpolate_θ(Fun_μ_hyp, MDEIM, timesθ)::Function
  else
    Fun_μ_hyp = assemble_hyperred_fun_mat(
      FEMSpace, FEMInfo, Param, MDEIM.el, timesθ)
    MDEIM_online(Fun_μ_hyp, MDEIM.Matᵢ, MDEIM.idx, length(timesθ))::Function
  end

end

function θ_function(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  Param::ParamInfoST,
  MDEIM::VMDEIM{T}) where {ID,D,T}

  @assert isnonlinear(RBInfo, Param.var) "This method is only for nonlinear variables"

  if RBInfo.st_mdeim
    red_timesθ = timesθ[MDEIM.time_idx]
    Fun_μ_hyp = assemble_hyperred_fun_vec(
      FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ)
    interpolate_θ(Fun_μ_hyp, MDEIM, timesθ)::Function
  else
    Fun_μ_hyp = assemble_hyperred_fun_vec(
      FEMSpace, FEMInfo, Param, MDEIM.el, timesθ)
    MDEIM_online(Fun_μ_hyp, MDEIM.Matᵢ, MDEIM.idx, length(timesθ))::Function
  end

end

function MDEIM_online(
  Mat_nonaffine::AbstractArray{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  θvec = Matᵢ \ reshape(Mat_nonaffine, :, Nₜ)[idx,:]
  blocks(Matrix{T}(θvec'))

end

function MDEIM_online(
  Mats_nonaffine::Vector{<:AbstractArray{T}},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  Mat_nonaffine = Matrix(Mats_nonaffine)
  MDEIM_online(Mat_nonaffine, Matᵢ, idx, Nₜ)

end

function MDEIM_online(
  Fun_nonaffine::Function,
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  θmat(u) = Matᵢ \ reshape(Fun_nonaffine(u), :, Nₜ)[idx,:]
  θblock(u) = blocks(Matrix{T}(θmat(u)'))

  θblock

end
