function get_Aₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB)

  get_Aₙ(RBInfo, RBVars.S)

end

function get_Mₙ(
  RBInfo::Info,
  RBVars::PoissonSTPGRB{T}) where T

  if (isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv")) &&
      isfile(joinpath(RBInfo.paths.ROM_structures_path, "MΦ.csv")) &&
      isfile(joinpath(RBInfo.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv")))
    println("Importing reduced affine stiffness matrix")
    Mₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,:)
    Qᵐ = sqrt(size(RBVars.Mₙ)[end])
    @assert floor(Qᵐ) == Qᵐ "Qᵐ should be the square root of an Int64"
    RBVars.Qᵐ = Int(Qᵐ)
    MΦ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.ROM_structures_path, "MΦ.csv"))
    RBVars.MΦ = reshape(MΦ,RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,:)
    MΦᵀPᵤ⁻¹ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    RBVars.MΦᵀPᵤ⁻¹ = reshape(MΦᵀPᵤ⁻¹,RBVars.S.nₛᵘ,RBVars.S.Nₛᵘ,:)
    return []
  else
    println("Failed to import the reduced affine mass matrix: must build it")
    return ["M"]
  end

end

function get_MAₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T}) where T

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    println("S-PGRB: importing MAₙ")
    MAₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
    RBVars.MAₙ = reshape(MAₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,:)
    return []
  else
    println("ST-PGRB: failed to import MAₙ: must build it")
    return ["M","A"]
  end

end

function assemble_MAₙ(RBVars::PoissonSTPGRB{T}) where T

  println("S-PGRB: Assembling MAₙ")

  nₛᵘ = RBVars.S.nₛᵘ
  MAₙ = zeros(T,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,RBVars.Qᵐ*RBVars.S.Qᵃ)
  MΦᵀ = permutedims(RBVars.MΦ,[2,1,3])
  AΦᵀPᵤ⁻¹ᵀ = permutedims(RBVars.S.AΦᵀPᵤ⁻¹,[2,1,3])
  [MAₙ[:,:,(qᵃ-1)*RBVars.Qᵐ+qᵐ] = MΦᵀ[:,:,qᵐ]*AΦᵀPᵤ⁻¹ᵀ[:,:,qᵃ]
    for qᵐ=1:RBVars.Qᵐ for qᵃ=1:RBVars.S.Qᵃ]
  RBVars.MAₙ = MAₙ

end

function assemble_affine_matrices(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  var::String) where T

  get_inverse_P_matrix(RBInfo, RBVars.S)

  if var == "M"
    RBVars.Qᵐ = 1
    println("Assembling affine reduced mass")
    M = load_CSV(sparse([],[],T[]), joinpath(RBInfo.paths.FEM_structures_path, "M.csv"))
    RBVars.Mₙ = reshape((M*RBVars.S.Φₛᵘ)'*RBVars.S.Pᵤ⁻¹*(M*RBVars.S.Φₛᵘ),
      RBVars.S.Nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ)
    RBVars.MΦ = reshape(M*RBVars.S.Φₛᵘ,
      RBVars.S.Nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ)
    RBVars.MΦᵀPᵤ⁻¹ = reshape((M*RBVars.S.Φₛᵘ)'*Matrix(RBVars.S.Pᵤ⁻¹),
      RBVars.S.Nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ)
  else
    assemble_affine_matrices(RBInfo, RBVars.S, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  MDEIM_mat::Matrix,
  row_idx::Vector{Int64},
  var::String) where T

  if RBInfo.space_time_M_DEIM
    Nₜ = RBVars.Nₜ
    MDEIM_mat_new = reshape(MDEIM_mat,length(row_idx),RBVars.Nₜ,:)
    Q = size(MDEIM_mat_new)[3]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.S.Nₛᵘ)
    MatqΦ = zeros(T,RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,Q*Nₜ)
    for q = 1:Q
      println("ST-GRB: affine component number $q/$Q, matrix $var")
      for j = 1:RBVars.S.Nₛᵘ
        Mat_idx = findall(x -> x == j, r_idx)
        MatqΦ[j,:,(q-1)*Nₜ+1:q*Nₜ] =
          (MDEIM_mat_new[Mat_idx,:,q]' * RBVars.S.Φₛᵘ[c_idx[Mat_idx],:])'
      end
    end
    MatqΦᵀPᵤ⁻¹ = zeros(T,RBVars.S.nₛᵘ,RBVars.S.Nₛᵘ,Q*Nₜ)
    matrix_product!(MatqΦᵀPᵤ⁻¹,MatqΦ,Matrix(RBVars.S.Pᵤ⁻¹),transpose_A=true)
    for q₁ = 1:Q*Nₜ
      for q₂ = 1:Q*Nₜ
        println("ST-PGRB: affine component number $((q₁-1)*RBVars.Qᵐ+q₂), matrix $var")
        RBVars.Mₙ[:,:,(q₁-1)*Q+q₂] = MatqΦᵀPᵤ⁻¹[:,:,q₁]*MatqΦ[:,:,q₂]
      end
    end
  else
    Q = size(MDEIM_mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.S.Nₛᵘ)
    MatqΦ = zeros(T,RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,Q)
    for j = 1:RBVars.S.Nₛᵘ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.S.Φₛᵘ[c_idx[Mat_idx],:])'
    end
    MatqΦᵀPᵤ⁻¹ = zeros(T,RBVars.S.nₛᵘ,RBVars.S.Nₛᵘ,Q)
    matrix_product!(MatqΦᵀPᵤ⁻¹,MatqΦ,Matrix(RBVars.S.Pᵤ⁻¹),transpose_A=true)
    Matₙ = zeros(T,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,Q^2)
    for q₁ = 1:Q
      for q₂ = 1:Q
        println("ST-PGRB: affine component number $((q₁-1)*RBVars.Qᵐ+q₂), matrix $var")
        Matₙ[:,:,(q₁-1)*Q+q₂] = MatqΦᵀPᵤ⁻¹[:,:,q₁]*MatqΦ[:,:,q₂]
      end
    end
  end

  if var == "M"
    RBVars.Mₙ = Matₙ
    RBVars.MΦᵀPᵤ⁻¹ = MatqΦᵀPᵤ⁻¹
    RBVars.Qᵐ = Q
  else
    RBVars.S.Aₙ = Matₙ
    RBVars.S.AΦᵀPᵤ⁻¹ = MatqΦᵀPᵤ⁻¹
    RBVars.S.Qᵃ = Q
  end

end

function assemble_affine_vectors(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  var::String) where T

  println("SPGRB: assembling affine reduced RHS; A is affine")

  if var == "F"
    RBVars.S.Qᶠ = 1
    println("Assembling affine reduced forcing term")
    F = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
    MFₙ = zeros(T, RBVars.S.nₛᵘ, 1, RBVars.Qᵐ*RBVars.S.Qᶠ)
    matrix_product_vec!(MFₙ, RBVars.MΦᵀPᵤ⁻¹, reshape(F,:,1))
    AFₙ = zeros(T, RBVars.S.nₛᵘ, 1, RBVars.S.Qᵃ*RBVars.S.Qᶠ)
    matrix_product_vec!(AFₙ, RBVars.S.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RBVars.S.Fₙ = hcat(reshape(MFₙ,:,RBVars.Qᵐ*RBVars.S.Qᶠ),
      reshape(AFₙ,:,RBVars.S.Qᵃ*RBVars.S.Qᶠ))
  elseif var == "H"
    RBVars.S.Qʰ = 1
    println("Assembling affine reduced Neumann term")
    H = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.FEM_structures_path, "H.csv"))
    MHₙ = zeros(T, RBVars.S.nₛᵘ, 1, RBVars.Qᵐ*RBVars.S.Qʰ)
    matrix_product_vec!(MHₙ, RBVars.MΦᵀPᵤ⁻¹, reshape(H,:,1))
    AHₙ = zeros(T, RBVars.S.nₛᵘ, 1, RBVars.S.Qᵃ*RBVars.S.Qʰ)
    matrix_product_vec!(AHₙ, RBVars.S.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RBVars.S.Hₙ = hcat(reshape(MHₙ,:,RBVars.Qᵐ*RBVars.S.Qʰ),
      reshape(AHₙ,:,RBVars.S.Qᵃ*RBVars.S.Qʰ))
  else
    error("Unrecognized variable to assemble")
  end

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  DEIM_mat::Matrix,
  var::String) where T

  if RBInfo.space_time_M_DEIM
    Nₜ = RBVars.Nₜ
    DEIM_mat_new = reshape(DEIM_mat,RBVars.S.Nₛᵘ,:)
    Q = Int(size(DEIM_mat_new)[2]/Nₜ)
    MVecₙ = zeros(T,RBVars.S.nₛᵘ,1,RBVars.Qᵐ*Q*Nₜ)
    matrix_product_vec!(MVecₙ,RBVars.MΦᵀPᵤ⁻¹,DEIM_mat_new)
    MVecₙ = reshape(MVecₙ,:,RBVars.Qᵐ*Q*Nₜ)
    AVecₙ = zeros(T,RBVars.S.nₛᵘ,1,RBVars.S.Qᵃ*Q*Nₜ)
    matrix_product_vec!(AVecₙ,RBVars.S.AΦᵀPᵤ⁻¹,DEIM_mat_new)
    AVecₙ = reshape(AVecₙ,:,RBVars.S.Qᵃ*Q*Nₜ)
    Vecₙ = hcat(MVecₙ,AVecₙ)
  else
    Q = size(DEIM_mat)[2]
    MVecₙ = zeros(T,RBVars.S.nₛᵘ,1,RBVars.Qᵐ*Q)
    matrix_product_vec!(MVecₙ,RBVars.MΦᵀPᵤ⁻¹,DEIM_mat)
    MVecₙ = reshape(MVecₙ,:,RBVars.Qᵐ*Q)
    AVecₙ = zeros(T,RBVars.S.nₛᵘ,1,RBVars.S.Qᵃ*Q)
    matrix_product_vec!(AVecₙ,RBVars.S.AΦᵀPᵤ⁻¹,DEIM_mat)
    AVecₙ = reshape(AVecₙ,:,RBVars.S.Qᵃ*Q)
    Vecₙ = hcat(MVecₙ,AVecₙ)
  end

  if var == "F"
    RBVars.S.Fₙ = Vecₙ
    RBVars.S.Qᶠ = Q
  elseif var == "H"
    RBVars.S.Hₙ = Vecₙ
    RBVars.S.Qʰ = Q
  else
    error("Unrecognized vector to assemble with DEIM")
  end

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB,
  operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.S.offline_time += @elapsed begin
    if "M" ∈ operators || "F" ∈ operators || "H" ∈ operators
      if !RBInfo.probl_nl["M"]
        assemble_affine_matrices(RBInfo, RBVars, "M")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "M")
      end
    end

    if "A" ∈ operators || "F" ∈ operators || "H" ∈ operators
      if !RBInfo.probl_nl["A"]
        assemble_affine_matrices(RBInfo, RBVars, "A")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "A")
      end
    end

    if "F" ∈ operators
      if !RBInfo.probl_nl["f"]
        assemble_affine_vectors(RBInfo, RBVars, "F")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "F")
      end
    end

    if "H" ∈ operators
      if !RBInfo.probl_nl["h"]
        assemble_affine_vectors(RBInfo, RBVars, "H")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "H")
      end
    end

    if "A" ∈ operators || "F" ∈ operators
      assemble_MAₙ(RBVars)
    end
  end

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSTPGRB)

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Mₙ,:,RBVars.Qᵐ^2),
      joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV(reshape(RBVars.MAₙ,:,RBVars.Qᵐ*RBVars.S.Qᵃ),
      joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
    save_CSV(reshape(RBVars.MΦᵀPᵤ⁻¹,:,RBVars.Qᵐ),
      joinpath(RBInfo.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    save_affine_structures(RBInfo, RBVars.S)
  end
end

function get_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSTPGRB)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars.S))
  append!(operators, get_Mₙ(RBInfo, RBVars))
  append!(operators, get_MAₙ(RBInfo, RBVars))
  return operators
end

function get_Q(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB)

  if RBVars.Qᵐ == 0
    Qᵐ = sqrt(size(RBVars.Mₙ)[end])
    @assert floor(Qᵐ) == Qᵐ "Qᵐ should be the square root of an Int64"
    RBVars.Qᵐ = Int(Qᵐ)
  end
  get_Q(RBInfo, RBVars.S)
end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵐᵃ::Matrix,
  θᵃᵐ::Matrix) where T

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  θ = RBInfo.θ
  δt = RBInfo.δt
  nₜᵘ = RBVars.nₜᵘ
  Qᵐᵃ = RBVars.Qᵐ*RBVars.S.Qᵃ
  Qᵐ = RBVars.Qᵐ^2
  Qᵃ = RBVars.S.Qᵃ^2

  Φₜᵘ_M = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ₋₁₋₁_M = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ₋₁₁_M = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)

  Φₜᵘ_A = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)
  Φₜᵘ₋₁₋₁_A = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)
  Φₜᵘ₋₁₁_A = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)

  Φₜᵘ_MA = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₋₁_MA = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₁₋₁_MA = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₁_MA = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)

  Φₜᵘ₁₋₁_AM = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₁_AM = zeros(T, RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:])
    for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:])
    for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_MA[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐᵃ[q,:])
    for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₋₁₋₁_M[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end])
    for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_A[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end])
    for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_MA[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐᵃ[q,2:end])
    for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₋₁₁_M[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵐ[q,2:end])
    for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_A[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵃ[q,2:end])
    for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_MA[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵐᵃ[q,2:end])
    for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_AM[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵃᵐ[q,2:end])
    for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₁₋₁_MA[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐᵃ[q,2:end])
    for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_AM[i_t,j_t,q] =
    sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃᵐ[q,2:end])
    for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  block₁ = zeros(T, RBVars.nᵘ, RBVars.nᵘ)

  for i_s = 1:RBVars.S.nₛᵘ

    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.S.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t, RBVars)

          term1 = RBVars.S.Aₙ[i_s,j_s,:]'*( θ^2*Φₜᵘ_A[i_t,j_t,:] +
            (1-θ)^2*θ^2*Φₜᵘ₋₁₋₁_A[i_t,j_t,:] + θ*(1-θ)*Φₜᵘ₋₁₁_A[i_t,j_t,:] +
            θ*(1-θ)*Φₜᵘ₋₁₁_A[j_t,i_t,:] )
          term2 = RBVars.MAₙ[i_s,j_s,:]'*( θ*Φₜᵘ_MA[i_t,j_t,:] -
            (1-θ)*Φₜᵘ₋₁₋₁_MA[i_t,j_t,:] - θ*Φₜᵘ₋₁₁_MA[i_t,j_t,:] +
            (1-θ)*Φₜᵘ₁₋₁_MA[i_t,j_t,:] )
          term3 = RBVars.MAₙ[j_s,i_s,:]'*( θ*Φₜᵘ_MA[j_t,i_t,:] -
            (1-θ)*Φₜᵘ₋₁₋₁_MA[j_t,i_t,:] + (1-θ)*Φₜᵘ₋₁₁_AM[i_t,j_t,:] -
            θ*Φₜᵘ₁₋₁_AM[i_t,j_t,:] )
          term4 = RBVars.Mₙ[i_s,j_s,:]'*( Φₜᵘ_M[i_t,j_t,:] +
            Φₜᵘ₋₁₋₁_M[i_t,j_t,:] - Φₜᵘ₋₁₁_M[i_t,j_t,:] -
            Φₜᵘ₋₁₁_M[j_t,i_t,:])

          block₁[i_st,j_st] = θ^2*(δt^2*term1 + δt*term2 + δt*term3 + term4)

        end
      end

    end
  end

  push!(RBVars.S.LHSₙ, block₁)

end

function get_RB_RHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  println("Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)")

  Qᵐᶠ = RBVars.Qᵐ*RBVars.S.Qᶠ
  Qᵐʰ = RBVars.Qᵐ*RBVars.S.Qʰ
  Qᶠ_tot = RBVars.S.Qᶠ*(RBVars.Qᵐ+RBVars.S.Qᵃ)
  Qʰ_tot = RBVars.S.Qʰ*(RBVars.Qᵐ+RBVars.S.Qᵃ)
  δt = RBInfo.δt
  θ = RBInfo.θ
  δtθ = δt*θ
  nₜᵘ = RBVars.nₜᵘ

  Φₜᵘ_F = zeros(T, RBVars.nₜᵘ, Qᶠ_tot)
  Φₜᵘ₋₁₁_F = zeros(T, RBVars.nₜᵘ, Qᶠ_tot)
  Φₜᵘ_H = zeros(T, RBVars.nₜᵘ, Qʰ_tot)
  Φₜᵘ₋₁₁_H = zeros(T, RBVars.nₜᵘ, Qʰ_tot)

  [Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:])
    for q = 1:Qᶠ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_F[i_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*θᶠ[q,2:end])
    for q = 1:Qᶠ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:])
    for q = 1:Qʰ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_H[i_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*θʰ[q,2:end])
    for q = 1:Qʰ_tot for i_t = 1:nₜᵘ]

  block₁ = zeros(T,RBVars.nᵘ,1)
  for i_s = 1:RBVars.S.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      term1 = (RBVars.S.Fₙ[i_s,1:Qᵐᶠ]'*Φₜᵘ_F[i_t,1:Qᵐᶠ] +
        RBVars.S.Hₙ[i_s,1:Qᵐʰ]'*Φₜᵘ_H[i_t,1:Qᵐʰ])
      term2 = δtθ*(RBVars.S.Fₙ[i_s,Qᵐᶠ+1:end]'*Φₜᵘ_F[i_t,Qᵐᶠ+1:end] +
        RBVars.S.Hₙ[i_s,Qᵐʰ+1:end]'*Φₜᵘ_H[i_t,Qᵐʰ+1:end])
      term3 = -θ*( RBVars.S.Fₙ[i_s,1:Qᵐᶠ]'*Φₜᵘ₋₁₁_F[i_t,1:Qᵐᶠ] +
        RBVars.S.Hₙ[i_s,1:Qᵐʰ]'*Φₜᵘ₋₁₁_H[i_t,1:Qᵐʰ] )
      term4 = δtθ*(1-θ)*(RBVars.S.Fₙ[i_s,Qᵐᶠ+1:end]'*Φₜᵘ₋₁₁_F[i_t,Qᵐᶠ+1:end] +
        RBVars.S.Hₙ[i_s,Qᵐʰ+1:end]'*Φₜᵘ₋₁₁_H[i_t,Qᵐʰ+1:end])

      block₁[i_st,1] = term1 + term2 + term3 + term4

    end
  end

  block₁ *= δtθ
  push!(RBVars.S.RHSₙ, block₁)

end

function get_RB_system(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB,
  Param::ParametricInfoUnsteady)

  initialize_RB_system(RBVars.S)
  initialize_online_time(RBVars.S)

  RBVars.S.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

    θᵐ, θᵐᵃ, θᵃᵐ, θᵃ, θᶠ, θʰ = get_θ(RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ, θᵐᵃ, θᵃᵐ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_Parametric_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
      else
        build_Param_RHS(RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function build_Param_RHS(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  Param::ParametricInfoUnsteady) where T

  println("Assembling RHS exactly using θ-method time scheme, θ=$(RBInfo.θ)")

  δt = RBInfo.δt
  θ = RBInfo.θ
  δtθ = δt*θ
  θᵐ_temp = get_θᵐ(RBInfo, RBVars, Param)
  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)

  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  F, H = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ), zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo.FEMInfo)
  for (i, tᵢ) in enumerate(timesθ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  RHS = (F+H)*δtθ

  MRHSΦₜ = zeros(T, RBVars.S.Nₛᵘ,RBVars.nₜᵘ,RBVars.Qᵐ)
  MRHSΦₜ₁ = zeros(T, RBVars.S.Nₛᵘ,RBVars.nₜᵘ,RBVars.Qᵐ)
  ARHSΦₜ = zeros(T, RBVars.S.Nₛᵘ,RBVars.nₜᵘ,RBVars.S.Qᵃ)
  ARHSΦₜ₁ = zeros(T, RBVars.S.Nₛᵘ,RBVars.nₜᵘ,RBVars.S.Qᵃ)
  for iₛ=1:RBVars.S.Nₛᵘ
    for jₜ=1:RBVars.nₜᵘ
      for qᵐ=1:RBVars.Qᵐ
        MRHSΦₜ[iₛ,jₜ,qᵐ] = sum(RHS[iₛ,:].*RBVars.Φₜᵘ[:,jₜ].*θᵐ_temp[qᵐ,:])
        MRHSΦₜ₁[iₛ,jₜ,qᵐ] = sum(RHS[iₛ,2:end].*RBVars.Φₜᵘ[1:end-1,jₜ].*θᵐ_temp[qᵐ,2:end])
      end
      for qᵃ=1:RBVars.S.Qᵃ
        ARHSΦₜ[iₛ,jₜ,qᵃ] = sum(RHS[iₛ,:].*RBVars.Φₜᵘ[:,jₜ].*θᵃ_temp[qᵃ,:])
        ARHSΦₜ₁[iₛ,jₜ,qᵃ] = sum(RHS[iₛ,2:end].*RBVars.Φₜᵘ[1:end-1,jₜ].*θᵃ_temp[qᵃ,2:end])
      end
    end
  end

  MRHSₙ = zeros(T,RBVars.S.nₛᵘ,RBVars.nₜᵘ)
  ARHSₙ = zeros(T,RBVars.S.nₛᵘ,RBVars.nₜᵘ)
  MRHS₁ₙ = zeros(T,RBVars.S.nₛᵘ,RBVars.nₜᵘ)
  ARHS₁ₙ = zeros(T,RBVars.S.nₛᵘ,RBVars.nₜᵘ)
  for qᵐ=1:RBVars.Qᵐ
    MRHSₙ += RBVars.MΦᵀPᵤ⁻¹[:,:,qᵐ]*MRHSΦₜ[:,:,qᵐ]
    MRHS₁ₙ += RBVars.MΦᵀPᵤ⁻¹[:,:,qᵐ]*MRHSΦₜ₁[:,:,qᵐ]
  end
  for qᵃ=1:RBVars.S.Qᵃ
    ARHSₙ += RBVars.S.AΦᵀPᵤ⁻¹[:,:,qᵃ]*ARHSΦₜ[:,:,qᵃ]
    ARHS₁ₙ += RBVars.S.AΦᵀPᵤ⁻¹[:,:,qᵃ]*ARHSΦₜ₁[:,:,qᵃ]
  end

  RHSₙ = MRHSₙ+δtθ*ARHSₙ + θ*(δt*(1-θ)*ARHS₁ₙ-MRHS₁ₙ)
  push!(RBVars.S.RHSₙ, reshape(RHSₙ',:,1))

end

function get_θ(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTPGRB{T},
  Param::ParametricInfoUnsteady) where T

  θᵐ_temp = get_θᵐ(RBInfo, RBVars, Param)
  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)
  Qᵐ, Qᵃ = size(θᵐ_temp)[1], size(θᵃ_temp)[1]

  θᵐ = zeros(T,Qᵐ^2,RBVars.Nₜ)
  θᵃ = zeros(T,Qᵃ^2,RBVars.Nₜ)
  θᵐᵃ = zeros(T,Qᵐ*Qᵃ,RBVars.Nₜ)
  θᵃᵐ = zeros(T,Qᵐ*Qᵃ,RBVars.Nₜ)

  for q₁ = 1:Qᵐ
    for q₂ = 1:Qᵐ
      θᵐ[(q₁-1)*Qᵐ+q₂,:] = θᵐ_temp[q₁,:].*θᵐ_temp[q₂,:]
    end
    for q₂ = 1:Qᵃ
      θᵐᵃ[(q₁-1)*Qᵃ+q₂,:] = θᵐ_temp[q₁,:].*θᵃ_temp[q₂,:]
    end
  end
  for q₁ = 1:Qᵃ
    for q₂ = 1:Qᵃ
      θᵃ[(q₁-1)*Qᵃ+q₂,:] = θᵃ_temp[q₁,:].*θᵃ_temp[q₂,:]
    end
    for q₂ = 1:Qᵐ
      θᵃᵐ[(q₁-1)*Qᵐ+q₂,:] = θᵐ_temp[q₂,:].*θᵃ_temp[q₁,:]
    end
  end

  if !RBInfo.build_Parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(RBInfo, RBVars, Param)
    Qᶠ, Qʰ = size(θᶠ_temp)[1], size(θʰ_temp)[1]
    θᵐᶠ = zeros(T,Qᵐ*Qᶠ,RBVars.Nₜ)
    θᵐʰ = zeros(T,Qᵐ*Qʰ,RBVars.Nₜ)
    θᵃᶠ = zeros(T,Qᵃ*Qᶠ,RBVars.Nₜ)
    θᵃʰ = zeros(T,Qᵃ*Qʰ,RBVars.Nₜ)

    for q₁ = 1:Qᵐ
      for q₂ = 1:Qᶠ
        θᵐᶠ[(q₁-1)*Qᶠ+q₂,:] = θᵐ_temp[q₁,:].*θᶠ_temp[q₂,:]
      end
      for q₂ = 1:Qʰ
        θᵐʰ[(q₁-1)*Qʰ+q₂,:] = θᵐ_temp[q₁,:].*θʰ_temp[q₂,:]
      end
    end
    for q₁ = 1:Qᵃ
      for q₂ = 1:Qᶠ
        θᵃᶠ[(q₁-1)*Qᶠ+q₂,:] = θᵃ_temp[q₁,:].*θᶠ_temp[q₂,:]
      end
      for q₂ = 1:Qʰ
        θᵃʰ[(q₁-1)*Qʰ+q₂,:] = θᵃ_temp[q₁,:].*θʰ_temp[q₂,:]
      end
    end

    θᶠ = vcat(θᵐᶠ, θᵃᶠ)
    θʰ = vcat(θᵐʰ, θᵃʰ)

  else

    θᶠ, θʰ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)

  end

  return θᵐ, θᵐᵃ, θᵃᵐ, θᵃ, θᶠ, θʰ

end
