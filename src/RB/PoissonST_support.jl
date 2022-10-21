################################# OFFLINE ######################################

POD_space(RBInfo::Info, RBVars::PoissonST) =
  POD_space(RBInfo, RBVars.Steady)

function POD_time(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T}) where T

  println("Spatial POD for field p, tolerance: $(RBInfo.ϵₛ)")

  if RBInfo.t_red_method == "ST-HOSVD"
    Sᵘ = RBVars.Φₛ' * RBVars.Sᵘ
  else
    Sᵘ = RBVars.Sᵘ
  end
  Sᵘₜ = mode₂_unfolding(Sᵘ, RBInfo.nₛ)

  Φₜᵘ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end

function index_mapping(
  i::Int,
  j::Int,
  RBVars::PoissonST)

  Int((i-1)*RBVars.nₜᵘ+j)

end

function set_operators(
  RBInfo::Info,
  RBVars::PoissonST)

  vcat(["M"], set_operators(RBInfo, RBVars.Steady))

end

function get_A(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_A(RBInfo, RBVars.Steady)

  if isempty(op)
    if "A" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_A.csv"))
        RBVars.MDEIM_A.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_A.csv"))
      else
        op = ["A"]
      end
    end
  end

  op

end

function get_M(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))

    Mₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}

    if "M" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_M.Matᵢ, RBVars.MDEIM_M.idx, RBVars.MDEIM_M.time_idx, RBVars.MDEIM_M.el) =
        load_structures_in_list(("Matᵢ_M", "idx_M", "time_idx_M", "el_M"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0),
        Vector{Int}(undef,0)), RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for M: must build them")
    op = ["M"]

  end

  op

end

function get_F(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_F(RBInfo, RBVars.Steady)

  if isempty(op)
    if "F" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_F.csv"))
        RBVars.MDEIM_F.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_F.csv"))
      else
        op = ["F"]
      end
    end
  end

  op

end

function get_H(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_H(RBInfo, RBVars.Steady)

  if isempty(op)
    if "H" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_H.csv"))
        RBVars.MDEIM_H.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_H.csv"))
      else
        op = ["H"]
      end
    end
  end

  op

end

function get_L(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_L(RBInfo, RBVars.Steady)

  if isempty(op)
    if "L" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_L.csv"))
        RBVars.MDEIM_L.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_L.csv"))
      else
        op = ["L"]
      end
    end
  end

  op

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::PoissonST{T},
  var::String) where T

  if var == "M"
    println("Assembling affine reduced M")
    M = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "M.csv"))
    RBVars.Mₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Mₙ[:,:,1] = (RBVars.Φₛ)' * M * RBVars.Φₛ
  else
    assemble_affine_structures(RBInfo, RBVars.Steady, var)
  end

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
  var::String)

  println("The variable $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_A.Mat)
      MDEIM_offline!(RBVars.MDEIM_A, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_A, var)
  elseif var == "M"
    if isempty(RBVars.MDEIM_M.Mat)
      MDEIM_offline!(RBVars.MDEIM_M, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_M, var)
  elseif var == "F"
    if isempty(RBVars.MDEIM_F.Mat)
      MDEIM_offline!(RBVars.MDEIM_F, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_F, var)
  elseif var == "H"
    if isempty(RBVars.MDEIM_H.Mat)
      MDEIM_offline!(RBVars.MDEIM_H, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_H, var)
  elseif var == "L"
    if isempty(RBVars.MDEIM_L.Mat)
      MDEIM_offline!(RBVars.MDEIM_L, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_L, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonST{T},
  MDEIM::MMDEIM,
  var::String) where T

  Q = size(MDEIM.Mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵘ)

  function assemble_ith_row(i::Int)
    Mat_idx = findall(x -> x == i, r_idx)
    Matrix(reshape((MDEIM.Mat[Mat_idx,:]' * RBVars.Φₛ[c_idx[Mat_idx],:])', 1, :))
  end

  VecMatΦ = Broadcasting(assemble_ith_row)(1:RBVars.Nₛᵘ)::Vector{Matrix{T}}
  MatqΦ = Matrix{T}(reduce(vcat, VecMatΦ))::Matrix{T}
  Matₙ = Matrix{T}(reshape(RBVars.Φₛ' * MatqΦ, RBVars.nₛᵘ, :, Q))::Matrix{T}

  if var == "M"
    RBVars.Mₙ = Matₙ
  else
    RBVars.Aₙ = Matₙ
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonST{T},
  MDEIM::VMDEIM,
  var::String) where T

  assemble_reduced_mat_MDEIM(RBVars.Steady, MDEIM, var)

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonST{T},
  operators::Vector{String}) where T

  affine_vars = (reshape(RBVars.Mₙ, RBVars.nₛᵘ ^ 2, :)::Matrix{T},)
  affine_names = ("Mₙ",)
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  MDEIM_vars = (
    RBVars.MDEIM_M.Matᵢ, RBVars.MDEIM_M.idx, RBVars.MDEIM_M.time_idx, RBVars.MDEIM_M.el,
    RBVars.MDEIM_A.time_idx, RBVars.MDEIM_F.time_idx, RBVars.MDEIM_H.time_idx,
    RBVars.MDEIM_L.time_idx)
  MDEIM_names = (
    "Matᵢ_M","idx_M","time_idx_M","el_M",
    "time_idx_A","time_idx_F","time_idx_H","time_idx_L")
  save_structures_in_list(MDEIM_vars, MDEIM_names, RBInfo.ROM_structures_path)

  save_assembled_structures(RBInfo, RBVars.Steady, operators)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::PoissonST,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Steady, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::PoissonST,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Steady, LHS_blocks, RHS_blocks, operators)

end

function get_θ_matrix(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::ParamInfoST,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")::Matrix{T}
  elseif var == "M"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.m, RBVars.MDEIM_M, "M")::Matrix{T}
  elseif var == "F"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.MDEIM_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.MDEIM_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_L, "L")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end


function assemble_param_RHS(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::ParamInfoST) where T

  println("Assembling RHS exactly using θ-method time scheme, θ=$(RBInfo.θ)")

  F_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  RHS = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)

  for (i,tᵢ) in enumerate(timesθ)
    RHS[:,i] = F_t(tᵢ) + H_t(tᵢ) - L_t(tᵢ)
  end

  RHSₙ = RBVars.Φₛ'*(RHS*RBVars.Φₜᵘ)
  push!(RBVars.RHSₙ, reshape(RHSₙ',:,1))::Vector{Matrix{T}}

end

function adaptive_loop_on_params(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  mean_uₕ_test::Matrix,
  mean_pointwise_err::Matrix,
  μ::Vector{Vector{T}},
  param_nbs,
  n_adaptive=nothing) where T

  if isnothing(n_adaptive)
    nₛᵘ_add = floor(Int,RBVars.nₛᵘ*0.1)
    nₜᵘ_add = floor(Int,RBVars.nₜᵘ*0.1)
    n_adaptive = maximum(hcat([1,1],[nₛᵘ_add,nₜᵘ_add]),dims=2)::Vector{Int}
  end

  println("Running adaptive cycle: adding $n_adaptive temporal and spatial bases,
    respectively")

  time_err = zeros(T, RBVars.Nₜ)
  space_err = zeros(T, RBVars.Nₛᵘ)
  for iₜ = 1:RBVars.Nₜ
    time_err[iₜ] = (norm(mean_pointwise_err[:,iₜ],RBVars.Xu₀) /
      norm(mean_uₕ_test[:,iₜ],RBVars.Xu₀))
  end
  for iₛ = 1:RBVars.Nₛᵘ
    space_err[iₛ] = norm(mean_pointwise_err[iₛ,:])/norm(mean_uₕ_test[iₛ,:])
  end
  ind_s = argmax(space_err,n_adaptive[1])
  ind_t = argmax(time_err,n_adaptive[2])

  if isempty(RBVars.Sᵘ)
    Sᵘ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]
  else
    Sᵘ = RBVars.Sᵘ
  end
  Sᵘ = reshape(sum(reshape(Sᵘ,RBVars.Nₛᵘ,RBVars.Nₜ,:),dims=3),RBVars.Nₛᵘ,:)

  Φₛ_new = Matrix{T}(qr(Sᵘ[:,ind_t]).Q)[:,1:n_adaptive[2]]
  Φₜᵘ_new = Matrix{T}(qr(Sᵘ[ind_s,:]').Q)[:,1:n_adaptive[1]]
  RBVars.nₛᵘ += n_adaptive[2]
  RBVars.nₜᵘ += n_adaptive[1]
  RBVars.nᵘ = RBVars.nₛᵘ*RBVars.nₜᵘ

  RBVars.Φₛ = Matrix{T}(qr(hcat(RBVars.Φₛ,Φₛ_new)).Q)[:,1:RBVars.nₛᵘ]
  RBVars.Φₜᵘ = Matrix{T}(qr(hcat(RBVars.Φₜᵘ,Φₜᵘ_new)).Q)[:,1:RBVars.nₜᵘ]
  RBInfo.save_offline = false
  assemble_offline_structures(RBInfo, RBVars)

  loop_on_params(FEMSpace,RBInfo,RBVars,μ,param_nbs)

end

################################################################################

include("PoissonS.jl")
include("PoissonST_support.jl")

################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T}) where T

  println("Importing the snapshot matrix for field u,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵘ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo),"uₕ.csv"),
    DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]

  RBVars.Sᵘ = Sᵘ
  RBVars.Nₛᵘ = size(Sᵘ)[1]
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ

  println("Dimension of the snapshot matrix for field u: $(size(Sᵘ))")

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::PoissonST)

  get_norm_matrix(RBInfo, RBVars.Steady)

end

function assemble_reduced_basis(
  RBInfo::ROMInfoST,
  RBVars::PoissonST)

  println("Building the space-time reduced basis for field u")

  RBVars.offline_time += @elapsed begin
    POD_space(RBInfo, RBVars)
    POD_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ

  if RBInfo.save_offline
    save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path, "Φₛ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.ROM_structures_path, "Φₜᵘ.csv"))
  end

  return

end

function get_reduced_basis(
  RBInfo::Info,
  RBVars::PoissonST{T}) where T

  get_reduced_basis(RBInfo, RBVars.Steady)

  println("Importing the temporal reduced basis for field u")
  RBVars.Φₜᵘ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₜᵘ.csv"))
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ

end

function get_offline_structures(
  RBInfo::ROMInfoST,
  RBVars::PoissonST)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))
  append!(operators, get_M(RBInfo, RBVars))

  if !RBInfo.online_RHS
    append!(operators, get_F(RBInfo, RBVars))
    append!(operators, get_H(RBInfo, RBVars))
    append!(operators, get_L(RBInfo, RBVars))
  end

  operators

end

function assemble_offline_structures(
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ setdiff(operators, RBInfo.probl_nl)
      assemble_affine_structures(RBInfo, RBVars, var)
    end

    for var ∈ intersect(operators, RBInfo.probl_nl)
      assemble_MDEIM_structures(RBInfo, RBVars, var)
    end
  end

  if RBInfo.save_offline
    save_assembled_structures(RBInfo, RBVars, operators)
  end

end

function offline_phase(
  RBInfo::ROMInfoST,
  RBVars::PoissonST)

  println("Offline phase of the RB solver, unsteady Poisson problem")

  RBVars.Nₜ = Int(RBInfo.tₗ / RBInfo.δt)

  if RBInfo.get_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    get_snapshots_success = true
  else
    get_snapshots_success = false
  end

  if RBInfo.get_offline_structures
    get_reduced_basis(RBInfo, RBVars)
    get_basis_success = true
  else
    get_basis_success = false
  end

  if !get_snapshots_success && !get_basis_success
    error("Impossible to assemble the reduced problem if neither
      the snapshots nor the bases can be loaded")
  end

  if get_snapshots_success && !get_basis_success
    println("Failed to import the reduced basis, building it via POD")
    assemble_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.get_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

################################## ONLINE ######################################

function get_θ(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::ParamInfoST) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵐ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "M")

  if !RBInfo.online_RHS
    θᶠ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "L")
  else
    θᶠ, θʰ, θˡ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end

  return θᵃ, θᵐ, θᶠ, θʰ, θˡ

end

#= n1 = 10
n2 = 5
A = rand(n1,n1);
Abk = matrix_to_blocks(A);
B = rand(n1,n2);
Bbk = matrix_to_blocks(B);
AB = zeros(n1,n1,n2);
for i = 1:n1
  for j = 1:n1
    for q = 1:n2
      AB[i,j,q] = sum(A[:,i] .* A[:,j] .* B[:,q])
    end
  end
end

prod_ijq(i,j,q) = sum(A[:,i] .* A[:,j] .* B[:,q])
m1(i,j) = Broadcasting(z->prod_ijq(i,j,z))(1:n2)
m2(i) = Broadcasting(z->m1(i,z))(1:n1)
myABbk = Broadcasting(m2)(1:n1)
myAB = blocks_to_matrix(myABbk)

function indx_map(i::Int)
  iₛ = 1+Int(floor((i-1)/n1))
  iₜ = i-(iₛ-1)*n1
  iₛ, iₜ
end
prod_iq(i,j,q) = sum(Abk[i] .* Abk[j] .* Bbk[q])
m11(i,j) = Broadcasting(z->prod_iq(i,j,z))(1:n2)
m22(i) = Broadcasting(z->m11(i,z))(1:n1)
myABbk1 = Broadcasting(m22)(1:n1)
myAB1 = blocks_to_matrix(myABbk1)

m111(i,q) = Broadcasting(j->prod_iq(i,j,q))(1:n1)
m222(q) = Broadcasting(z->m111(z,q))(1:n1)
myABbk2 = Broadcasting(m222)(1:n2)
myAB2 = blocks_to_matrix(myABbk2) =#

function get_RB_LHS_blocks(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  θᵐ::Matrix{T},
  θᵃ::Matrix{T}) where T

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.Qᵃ

  Φₜᵘ_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ₁_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ₁_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)

  @simd for i_t = 1:nₜᵘ
    for j_t = 1:nₜᵘ
      for q = 1:Qᵐ
        Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:])
        Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end])
      end
      for q = 1:Qᵃ
        Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:])
        Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end])
      end
    end
  end

  Mₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Mₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Aₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Aₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)

  @simd for qᵐ = 1:Qᵐ
    Mₙ_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ_M[:,:,qᵐ])::Matrix{T}
    Mₙ₁_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ₁_M[:,:,qᵐ])::Matrix{T}
  end
  @simd for qᵃ = 1:Qᵃ
    Aₙ_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ_A[:,:,qᵃ])::Matrix{T}
    Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])::Matrix{T}
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ) / (RBInfo.θ*RBInfo.δt)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ) / (RBInfo.θ*RBInfo.δt)
  Aₙ = reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  block₁ = RBInfo.θ*(Aₙ+Mₙ) + (1-RBInfo.θ)*Aₙ₁ - RBInfo.θ*Mₙ₁
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  θᶠ::Matrix{T},
  θʰ::Matrix{T},
  θˡ::Matrix{T}) where T

  println("Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)")

  Φₜᵘ_F = zeros(T, RBVars.nₜᵘ, RBVars.Qᶠ)
  Φₜᵘ_H = zeros(T, RBVars.nₜᵘ, RBVars.Qʰ)
  Φₜᵘ_L = zeros(T, RBVars.nₜᵘ, RBVars.Qˡ)

  @simd for i_t = 1:RBVars.nₜᵘ
    for q = 1:RBVars.Qᶠ
      Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:])
    end
    for q = 1:RBVars.Qʰ
      Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:])
    end
    for q = 1:RBVars.Qˡ
      Φₜᵘ_L[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θˡ[q,:])
    end
  end

  block₁ = zeros(T, RBVars.nᵘ, 1)
  @simd for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ
      i_st = index_mapping(i_s, i_t, RBVars)
      Fₙ_μ_i_j = RBVars.Fₙ[i_s,:]'*Φₜᵘ_F[i_t,:]
      Hₙ_μ_i_j = RBVars.Hₙ[i_s,:]'*Φₜᵘ_H[i_t,:]
      Lₙ_μ_i_j = RBVars.Lₙ[i_s,:]'*Φₜᵘ_L[i_t,:]
      block₁[i_st] = Fₙ_μ_i_j + Hₙ_μ_i_j - Lₙ_μ_i_j
    end
  end

  push!(RBVars.RHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
  Param::ParamInfoST)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = [1]

  RBVars.online_time = @elapsed begin

    operators = get_system_blocks(RBInfo,RBVars,blocks,blocks)

    θᵃ, θᵐ, θᶠ, θʰ, θˡ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ, θˡ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function solve_RB_system(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
  Param::ParamInfoST)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")

  RBVars.online_time += @elapsed begin
    @fastmath RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::PoissonST)

  println("Reconstructing FEM solution from the newly computed RB one")
  uₙ = reshape(RBVars.uₙ, (RBVars.nₜᵘ, RBVars.nₛᵘ))
  @fastmath RBVars.ũ = RBVars.Φₛ * (RBVars.Φₜᵘ * uₙ)'

end

function loop_on_params(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  μ::Vector{Vector{T}},
  param_nbs) where T

  H1_L2_err = zeros(T, length(param_nbs))
  mean_H1_err = zeros(T, RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(T, RBVars.nᵘ, length(param_nbs))
  mean_uₕ_test = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)

  for (i_nb, nb) in enumerate(param_nbs)
    println("\n")
    println("Considering parameter number: $nb/$(param_nbs[end])")

    Param = ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]

    mean_uₕ_test += uₕ_test

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    if i_nb > 1
      mean_online_time = RBVars.online_time/(length(param_nbs)-1)
      mean_reconstruction_time = reconstruction_time/(length(param_nbs)-1)
    end

    H1_err_nb, H1_L2_err_nb = compute_errors(
        RBVars, uₕ_test, RBVars.ũ, RBVars.Xu₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.ũ
    uₙ_μ[:, i_nb] = RBVars.uₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)")
  end
  return (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
    H1_L2_err,mean_online_time,mean_reconstruction_time)
end

function online_phase(
  RBInfo,
  RBVars::PoissonST,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady Poisson problem")

  FEMSpace, μ = get_FEMμ_info(RBInfo)

  get_norm_matrix(RBInfo, RBVars.Steady)
  (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
    mean_online_time,mean_reconstruction_time) =
    loop_on_params(FEMSpace, RBInfo, RBVars, μ, param_nbs)

  adapt_time = 0.
  if RBInfo.adaptivity
    adapt_time = @elapsed begin
      (ũ_μ,uₙ_μ,_,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
      H1_L2_err,_,_) =
      adaptive_loop_on_params(FEMSpace, RBInfo, RBVars, mean_uₕ_test,
      mean_pointwise_err, μ, param_nbs)
    end
  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  res_path = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_online
    println("Saving the results...")
    create_dir(res_path)
    save_CSV(ũ_μ, joinpath(res_path, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(res_path, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(res_path, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(res_path, "H1_err.csv"))
    save_CSV(H1_L2_err, joinpath(res_path, "H1L2_err.csv"))

    if RBInfo.get_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)
    CSV.write(joinpath(res_path, "times.csv"),times)
  end

  pass_to_pp = Dict("res_path"=>res_path,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>Float.(mean_pointwise_err))

  if RBInfo.post_process
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

end
