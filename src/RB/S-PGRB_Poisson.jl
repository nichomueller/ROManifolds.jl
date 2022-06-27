
function get_inverse_P_matrix(RBInfo::Info, RBVars::PoissonSPGRB)

  if RBInfo.use_norm_X
    if isempty(RBVars.Pᵤ⁻¹)
      println("S-PGRB: building the inverse of the diag preconditioner of the H1 norm matrix")
      if isfile(joinpath(RBInfo.paths.FEM_structures_path, "Pᵤ⁻¹.csv"))
        RBVars.Pᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Pᵤ⁻¹.csv"),
          true)
      else
        get_norm_matrix(RBInfo, RBVars)
        diag_Xᵘ₀ = Vector{Float64}(diag(RBVars.Xᵘ₀))
        RBVars.Pᵤ⁻¹ = spdiagm(1 ./ diag_Xᵘ₀)
        save_CSV(RBVars.Pᵤ⁻¹, joinpath(RBInfo.paths.FEM_structures_path, "Pᵤ⁻¹.csv"))
      end
    end
    RBVars.Pᵤ⁻¹ = Matrix{Float64}(RBVars.Pᵤ⁻¹)
  else
    RBVars.Pᵤ⁻¹ = I(RBVars.Nₛᵘ)
  end
end

function get_Aₙ(RBInfo::Info, RBVars::PoissonSPGRB) :: Vector

  if (isfile(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv")) &&
      isfile(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv")))
    println("Importing reduced affine stiffness matrix")
    Aₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)
    Qᵃ = sqrt(size(RBVars.Aₙ)[3])
    @assert floor(Qᵃ) == Qᵃ "Qᵃ should be the square root of an Int64"
    RBVars.Qᵃ = Int(Qᵃ)
    println("S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹")
    AΦᵀPᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RBVars.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)
    return []
  else
    println("Failed to import the reduced affine stiffness matrix: must build it")
    return ["A"]
  end

end

function get_AΦᵀPᵤ⁻¹(RBInfo::Info, RBVars::PoissonSPGRB)

  println("S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹")
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    AΦᵀPᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RBVars.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)
    return
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

function assemble_affine_matrices(RBInfo::Info, RBVars::PoissonSPGRB, var::String)

  get_inverse_P_matrix(RBInfo, RBVars)

  if var == "A"
    RBVars.Qᵃ = 1
    println("Assembling affine reduced stiffness")
    println("SPGRB: affine component number 1, matrix A")
    A = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "A.csv"), true)
    RBVars.AΦᵀPᵤ⁻¹ = reshape((A*RBVars.Φₛᵘ)'*RBVars.Pᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,1)
    RBVars.Aₙ = reshape(RBVars.AΦᵀPᵤ⁻¹*(A*RBVars.Φₛᵘ),RBVars.nₛᵘ,RBVars.nₛᵘ,1)
  else
    error("Unrecognized variable to load")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonSPGRB,
  MDEIM_mat::Matrix{Float64},
  row_idx::Vector{Float64})

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  MatqΦᵀPᵤ⁻¹ = zeros(RBVars.nₛᵘ,RBVars.Nₛᵘ,Q)
  matrix_product!(MatqΦᵀPᵤ⁻¹,MatqΦ,Matrix(RBVars.Pᵤ⁻¹),transpose_A=true)
  Matₙ = zeros(RBVars.nₛᵘ,RBVars.nₛᵘ,Q^2)
  @simd for q₁ = 1:Q
    for q₂ = 1:Q
      println("ST-PGRB: affine component number $((q₁-1)*RBVars.Qᵐ+q₂), matrix $var")
      Matₙ[:,:,(q₁-1)*Q+q₂] = MatqΦᵀPᵤ⁻¹[:,:,q₁]*MatqΦ[:,:,q₂]
    end
  end

  RBVars.Aₙ = Matₙ
  RBVars.AΦᵀPᵤ⁻¹ = MatqΦᵀPᵤ⁻¹
  RBVars.Qᵃ = Q

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonSTPGRB,
  DEIM_mat::Matrix{Float64},
  var::String)

  Q = size(DEIM_mat)[2]
  MVecₙ = zeros(RBVars.nₛᵘ,1,RBVars.Qᵐ*Q)
  matrix_product_vec!(MVecₙ,RBVars.MΦᵀPᵤ⁻¹,DEIM_mat)
  MVecₙ = reshape(MVecₙ,:,RBVars.Qᵐ*Q)
  AVecₙ = zeros(RBVars.nₛᵘ,1,RBVars.Qᵐ*Q)
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

function assemble_affine_vectors(RBInfo::Info, RBVars::PoissonSPGRB, var::String)

  println("S-PGRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots")

  if var == "F"
    RBVars.Qᶠ = 1
    println("Assembling affine reduced forcing term")
    F = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
    Fₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qᶠ)
    matrix_product_vec!(Fₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RBVars.Fₙ = reshape(Fₙ,:,RBVars.Qᵃ*RBVars.Qᶠ)
  elseif var == "H"
    RBVars.Qʰ = 1
    println("Assembling affine reduced Neumann term")
    H = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "H.csv"))
    Hₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qʰ)
    matrix_product_vec!(Hₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RBVars.Hₙ = reshape(Hₙ,:,RBVars.Qᵃ*RBVars.Qʰ)
  else
    error("Unrecognized variable to assemble")
  end

end

function save_affine_structures(RBInfo::Info, RBVars::PoissonSPGRB)

  if RBInfo.save_offline_structures

    Aₙ = reshape(RBVars.Aₙ, :, RBVars.Qᵃ)
    AΦᵀPᵤ⁻¹ = reshape(RBVars.AΦᵀPᵤ⁻¹, :, RBVars.Qᵃ)
    save_CSV(Aₙ, joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    save_CSV(AΦᵀPᵤ⁻¹, joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))

    if !RBInfo.build_Parametric_RHS
      save_CSV(RBVars.Fₙ, joinpath(RBInfo.paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV(RBVars.Hₙ, joinpath(RBInfo.paths.ROM_structures_path, "Hₙ.csv"))
    end

  end

end

function get_Q(RBInfo::Info, RBVars::PoissonSPGRB)
  if RBVars.Qᵃ == 0
    Qᵃ = sqrt(size(RBVars.Aₙ)[end])
    @assert floor(Qᵃ) == Qᵃ "Qᵃ should be the square root of an Int64"
    RBVars.Qᵃ = Int(Qᵃ)
  end
  if !RBInfo.build_Parametric_RHS
    if RBVars.Qᶠ == 0
      RBVars.Qᶠ = Int(size(RBVars.Fₙ)[end]/RBVars.Qᵃ)
    end
    if RBVars.Qʰ == 0
      RBVars.Qʰ = Int(size(RBVars.Hₙ)[end]/RBVars.Qᵃ)
    end
  end
end

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSPGRB, Param, θᵃ::Vector{Float64}) :: Tuple

  θᵃ_temp = θᵃ[1:RBVars.Qᵃ]/sqrt(θᵃ[1])
  F = assemble_forcing(FEMSpace, RBInfo, Param)
  H = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RBVars.AΦᵀPᵤ⁻¹)
  Fₙ, Hₙ = AΦᵀPᵤ⁻¹ * F, AΦᵀPᵤ⁻¹ * H

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(RBInfo::Info, RBVars::PoissonSPGRB, Param) :: Tuple

  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)
  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:RBVars.Qᵃ for q₂ = 1:RBVars.Qᵃ]

  if !RBInfo.build_Parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(RBInfo, RBVars, Param)
    θᶠ = [θᵃ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:RBVars.Qᵃ for q₂ = 1:RBVars.Qᶠ]
    θʰ = [θᵃ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:RBVars.Qᵃ for q₂ = 1:RBVars.Qʰ]

  else

    θᶠ, θʰ = Float64[], Float64[]

  end

  θᵃ, θᶠ, θʰ

end
