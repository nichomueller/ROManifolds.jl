
function get_inverse_P_matrix(
  RBInfo::Info,
  RBVars::PoissonSPGRB{T}) where T

  if RBInfo.use_norm_X
    if isempty(RBVars.Pᵤ⁻¹)
      println("S-PGRB: building the inverse of the diag preconditioner of the H1 norm matrix")
      if isfile(joinpath(get_FEM_structures_path(RBInfo), "Pᵤ⁻¹.csv"))
        RBVars.Pᵤ⁻¹ = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "Pᵤ⁻¹.csv"))
      else
        get_norm_matrix(RBInfo, RBVars)
        diag_Xᵘ₀ = Vector{T}(diag(RBVars.Xᵘ₀))
        RBVars.Pᵤ⁻¹ = spdiagm(1 ./ diag_Xᵘ₀)
        save_CSV(RBVars.Pᵤ⁻¹, joinpath(get_FEM_structures_path(RBInfo), "Pᵤ⁻¹.csv"))
      end
    end
    RBVars.Pᵤ⁻¹ = Matrix{T}(RBVars.Pᵤ⁻¹)
  else
    RBVars.Pᵤ⁻¹ = I(RBVars.Nₛᵘ)
  end
end

function get_Aₙ(
  RBInfo::Info,
  RBVars::PoissonSPGRB{T}) where T

  if (isfile(joinpath(RBInfo.Paths.ROM_structures_path, "Aₙ.csv")) &&
      isfile(joinpath(RBInfo.Paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv")))
    println("Importing reduced affine stiffness matrix")
    Aₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
    Qᵃ = sqrt(size(RBVars.Aₙ)[3])
    @assert floor(Qᵃ) == Qᵃ "Qᵃ should be the square root of an Int"
    RBVars.Qᵃ = Int(Qᵃ)
    println("S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹")
    AΦᵀPᵤ⁻¹ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RBVars.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)::Array{T,3}
    return [""]
  else
    println("Failed to import the reduced affine stiffness matrix: must build it")
    return ["A"]
  end

end

function get_AΦᵀPᵤ⁻¹(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSPGRB{T}) where T

  println("S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹")
  if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    AΦᵀPᵤ⁻¹ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RBVars.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)
  else
    if !RBInfo.probl_nl["A"]
      println("S-PGRB: failed to build AΦᵀPᵤ⁻¹; have to assemble affine stiffness")
      assemble_affine_matrices(RBInfo, RBVars, "A")
    else
      println("S-PGRB: failed to build AΦᵀPᵤ⁻¹; have to assemble non-affine stiffness ")
      assemble_MDEIM_matrices(RBInfo, RBVars, "A")
    end
  end

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::PoissonSPGRB{T},
  var::String) where T

  get_inverse_P_matrix(RBInfo, RBVars)

  if var == "A"
    RBVars.Qᵃ = 1
    println("Assembling affine reduced stiffness")
    println("SPGRB: affine component number 1, matrix A")
    A = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "A.csv"))
    RBVars.Aₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Aₙ[:,:,1] = (RBVars.Φₛᵘ)' * A * RBVars.Φₛᵘ
    RBVars.AΦᵀPᵤ⁻¹ = zeros(T, RBVars.nₛᵘ, RBVars.Nₛᵘ, 1)
    RBVars.AΦᵀPᵤ⁻¹[:,:,1] = (A*RBVars.Φₛᵘ)' * RBVars.Pᵤ⁻¹
  else
    error("Unrecognized variable to load")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonSPGRB{T},
  MDEIM_mat::Matrix,
  row_idx::Vector{Int}) where T

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(T,RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  MatqΦᵀPᵤ⁻¹ = zeros(T,RBVars.nₛᵘ,RBVars.Nₛᵘ,Q)
  matrix_product!(MatqΦᵀPᵤ⁻¹,MatqΦ,Matrix(RBVars.Pᵤ⁻¹),transpose_A=true)
  Matₙ = zeros(T,RBVars.nₛᵘ,RBVars.nₛᵘ,Q^2)
  @simd for q₁ = 1:Q
    for q₂ = 1:Q
      println("ST-PGRB: affine component number $((q₁-1)*RBVars.Qᵐ+q₂), matrix $var")
      Matₙ[:,:,(q₁-1)*Q+q₂] = MatqΦᵀPᵤ⁻¹[:,:,q₁]*MatqΦ[:,:,q₂]
    end
  end

  RBVars.Aₙ = Matₙ::Array{T,3}
  RBVars.AΦᵀPᵤ⁻¹ = MatqΦᵀPᵤ⁻¹::Array{T,3}
  RBVars.Qᵃ = Q

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonSPGRB{T},
  DEIM_mat::Matrix,
  var::String) where T

  Q = size(DEIM_mat)[2]
  MVecₙ = zeros(T,RBVars.nₛᵘ,1,RBVars.Qᵐ*Q)
  matrix_product_vec!(MVecₙ,RBVars.MΦᵀPᵤ⁻¹,DEIM_mat)
  MVecₙ = reshape(MVecₙ,:,RBVars.Qᵐ*Q)
  AVecₙ = zeros(T,RBVars.nₛᵘ,1,RBVars.Qᵐ*Q)
  matrix_product_vec!(AVecₙ,RBVars.AΦᵀPᵤ⁻¹,DEIM_mat)
  AVecₙ = reshape(AVecₙ,:,RBVars.Qᵃ*Q)
  Vecₙ = hcat(MVecₙ,AVecₙ)

  if var == "F"
    RBVars.Fₙ = Vecₙ
    RBVars.Qᶠ = Q
  else var == "H"
    RBVars.Hₙ = Vecₙ
    RBVars.Qʰ = Q
  end

end

function assemble_affine_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSPGRB{T},
  var::String) where T

  println("S-PGRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots")

  if var == "F"
    RBVars.Qᶠ = 1
    println("Assembling affine reduced forcing term")
    F = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "F.csv"))
    Fₙ = zeros(T,RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qᶠ)
    matrix_product_vec!(Fₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RBVars.Fₙ = reshape(Fₙ,:,RBVars.Qᵃ*RBVars.Qᶠ)
  elseif var == "H"
    RBVars.Qʰ = 1
    println("Assembling affine reduced Neumann term")
    H = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "H.csv"))
    Hₙ = zeros(T,RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qʰ)
    matrix_product_vec!(Hₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RBVars.Hₙ = reshape(Hₙ,:,RBVars.Qᵃ*RBVars.Qʰ)
  else
    error("Unrecognized variable to assemble")
  end

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSPGRB{T}) where T

  if RBInfo.save_offline_structures

    save_CSV(reshape(RBVars.Aₙ, :, RBVars.Qᵃ)::Matrix{T},
      joinpath(RBInfo.Paths.ROM_structures_path, "Aₙ.csv"))
    save_CSV(reshape(RBVars.AΦᵀPᵤ⁻¹, :, RBVars.Qᵃ)::Matrix{T},
      joinpath(RBInfo.Paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))

    if !RBInfo.build_parametric_RHS
      save_CSV(RBVars.Fₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV(RBVars.Hₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Hₙ.csv"))
    end

  end

end

function get_Q(
  RBInfo::Info,
  RBVars::PoissonSPGRB)

  if RBVars.Qᵃ == 0
    Qᵃ = sqrt(size(RBVars.Aₙ)[end])
    @assert floor(Qᵃ) == Qᵃ "Qᵃ should be the square root of an Int"
    RBVars.Qᵃ = Int(Qᵃ)
  end
  if !RBInfo.build_parametric_RHS
    if RBVars.Qᶠ == 0
      RBVars.Qᶠ = Int(size(RBVars.Fₙ)[end]/RBVars.Qᵃ)
    end
    if RBVars.Qʰ == 0
      RBVars.Qʰ = Int(size(RBVars.Hₙ)[end]/RBVars.Qᵃ)
    end
  end

  return

end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSPGRB,
  Param::SteadyParametricInfo)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

    θᵃ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param, θᵃ)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function build_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSPGRB,
  Param::SteadyParametricInfo,
  θᵃ::Vector)

  θᵃ_temp = θᵃ[1:RBVars.Qᵃ]/sqrt(θᵃ[1])
  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  AΦᵀPᵤ⁻¹ = assemble_parametric_structure(θᵃ_temp, RBVars.AΦᵀPᵤ⁻¹)
  Fₙ, Hₙ = AΦᵀPᵤ⁻¹ * F, AΦᵀPᵤ⁻¹ * H

  push!(RBVars.RHSₙ, reshape(Fₙ'+Hₙ',:,1))::Vector{Matrix{T}}

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSPGRB,
  Param::SteadyParametricInfo)

  θᵃ_temp = get_θᵃ(FEMSpace, RBInfo, RBVars, Param)
  θᵃ = zeros(T, RBVars.Qᵃ^2, 1)
  for q₁ = 1:RBVars.Qᵃ
    for q₂ = 1:RBVars.Qᵃ
      θᵃ[(q₁-1)*RBVars.Qᵃ+q₂] = θᵃ_temp[q₁]*θᵃ_temp[q₂]
    end
  end

  if !RBInfo.build_parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(FEMSpace, RBInfo, RBVars, Param)
    θᶠ = zeros(T, RBVars.Qᵃ*RBVars.Qᶠ, 1)
    θʰ = zeros(T, RBVars.Qᵃ*RBVars.Qʰ, 1)
    for q₁ = 1:RBVars.Qᵃ
      for q₂ = 1:RBVars.Qᶠ
        θᶠ[(q₁-1)*RBVars.Qᵃ+q₂] = θᵃ_temp[q₁]*θᶠ_temp[q₂]
      end
      for q₂ = 1:RBVars.Qʰ
        θʰ[(q₁-1)*RBVars.Qᵃ+q₂] = θᵃ_temp[q₁]*θʰ_temp[q₂]
      end
    end

  else

    θᶠ, θʰ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)

  end

  θᵃ, θᶠ, θʰ

end
