mutable struct info
  Nₕ²::Int64; Nₜ::Int64; nₛ::Int64; ϵ::Float64;
end

function simple_model()

  L = 2
  D = 2
  n = 4

  function stretching(x::Point)
    m = zeros(length(x))
    m[1] = x[1]^2
    for i in 2:D
      m[i] = x[i]
    end
    Point(m)
  end

  pmin = Point(Fill(0,D))
  pmax = Point(Fill(L,D))
  partition = Tuple(Fill(n,D))
  model = CartesianDiscreteModel(pmin,pmax,partition,map=stretching)

  degree = 2
  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model)
  dΓn = Measure(Γn, degree)
  Γd = BoundaryTriangulation(model)
  dΓd = Measure(Γd, degree)
  Qₕ = CellQuadrature(Ω, degree)

  refFE = Gridap.ReferenceFE(lagrangian, Float64, 1)
  V₀ = TestFEMSpace(model, refFE; conformity=:H1)
  g₀(x, t::Real) = 0
  g₀(t::Real) = x->g₀(x,t)
  V = TransientTrialFEMSpace(V₀, g₀)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  FEMSpace = FEMSpacePoissonUnsteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FEMSpace

end

function MDEIM_offline_spacetime(II::info,S::Matrix,row_idx) :: Tuple

  MDEIM_mat, Σ = POD(S, II.ϵ)
  MDEIM_mat, MDEIM_idx, _ = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(row_idx,MDEIM_idx,FEMSpace.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse,FEMSpace.Nₛᵘ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  return MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, el

end

function get_θₛₜ(MDEIM_idx,MDEIMᵢ_mat,online_snap) ::Vector

  f = online_snap[MDEIM_idx]
  θ = MDEIMᵢ_mat\f

  return θ

end

function assign_label_to_sorted_elems(v::Vector)
  vnew = copy(v)
  vnew = sort(vnew)
  unique!(vnew)
  vret = Int.(indexin(v,vnew))
  return vret
end

function modify_MDEIM_idx(MDEIM_idx) ::Vector

  idx_space, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx,FEMSpace.Nₛᵘ^2)
  idx_time_new = assign_label_to_sorted_elems(idx_time)
  MDEIM_idx_new = (idx_time_new.-1)*FEMSpace.Nₛᵘ^2+idx_space

  return MDEIM_idx_new

end

function test_st_MDEIM(FEMSpace,II::info,S::Matrix,row_idx)

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, el = MDEIM_offline_spacetime(II,S,row_idx)
  #online_snap = define_online_ST_stiffness(FEMSpace, RBInfo, el)
  online_snap_fast = define_online_ST_stiffness_fast(FEMSpace, RBInfo, el, MDEIM_idx_sparse)
  MDEIM_idx_new = modify_MDEIM_idx(MDEIM_idx_sparse)
  #θ = get_θₛₜ(MDEIM_idx_sparse,MDEIMᵢ_mat,online_snap)
  θ_fast = get_θₛₜ(MDEIM_idx_new,MDEIMᵢ_mat,online_snap_fast)
  #@test θ ≈ θfast
  approx_snap = MDEIM_mat*θ_fast

  return reshape(approx_snap,:,1), size(MDEIM_mat)[2], el

end

#= function check_similarity_approx_matrices()
  approx_snap = MDEIM_mat*θfast
  _, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse, FEMSpace.Nₛᵘ^2)
  idx_time = sort(idx_time)
  unique!(idx_time)
  idx_collect = Int64[]
  for i = 1:length(idx_time)
    append!(idx_collect,collect(II.Nₕ²*(idx_time[i]-1)+1:II.Nₕ²*idx_time[i]))
  end
  approx_snap_red = MDEIM_mat[idx_collect,:]*θfast
end =#

function check_eigen(online_snap,approx_snap,Nₜ)

  resh_snap=reshape(online_snap,:,Nₜ)
  resh_approx=reshape(approx_snap,:,Nₜ)
  eig_snap_time,_ = eigs(resh_snap'*resh_snap;nev=Nₜ-1)
  eig_snap_time = sqrt.(eig_snap_time)
  eig_approx_time,_ = eigs(resh_approx'*resh_approx;nev=Nₜ-1)
  eig_approx_time = sqrt.(eig_approx_time)

  return reshape(eig_snap_time,:,1), reshape(eig_approx_time,:,1)

end

"""Unsteady case"""
function run_one_test_ST_MDEIM(FEMSpace, RBInfo, nₛMDEIM::Int64)

  snaps, row_idx = define_offline_snaps(FEMSpace, RBInfo, nₛMDEIM)
  #snaps_red = define_offline_snaps_red(snaps)
  II = info(length(row_idx), 50, nₛMDEIM, 1e-5)

  runtime = @elapsed begin
    approx_snap, Q, el = test_st_MDEIM(FEMSpace,II,snaps,row_idx)
  end
  #= runtime_red = @elapsed begin
    approx_snap_red, Q_red = test_st_MDEIM(FEMSpace,II,snaps_red,row_idx)
  end =#
  online_snap = define_online_ST_stiffness_old(FEMSpace, RBInfo, el, row_idx)
  diff = online_snap - approx_snap
  #diff_red = online_snap - approx_snap_red
  #err = [norm(diff)/norm(online_snap),norm(diff_red)/norm(online_snap)]
  err = norm(diff)/norm(online_snap)
  err,runtime,Q
end

function define_offline_snaps(FEMSpace, RBInfo, nₛMDEIM::Int64)

  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  Nₜ = length(timesθ)
  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  μ = load_CSV(path)

  for k = 1:nₛMDEIM
    println("Considering Parameter number $k, need $(nₛMDEIM-k) more!")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
    A_t = assemble_stiffness(FEMSpace, RBInfo, Param)

    for i_t = 1:Nₜ
      println("Snapshot at time step $i_t/Nₜ")
      A_i = A_t(timesθ[i_t])
      i, v = findnz(A_i[:])
      if i_t === 1
        global row_idx = i
        global snapsₖ = zeros(length(row_idx)*Nₜ)
      end
      global snapsₖ[(i_t-1)*length(row_idx)+1:i_t*length(row_idx)] = v
    end

    if k === 1
      global snaps = snapsₖ
    else
      global snaps = hcat(snaps, snapsₖ)
    end

  end

  return snaps, row_idx

end

function define_offline_snaps_red(full_snap)

  Q = size(full_snap)[2]

  #= #test1
  snap,_ = POD(full_snap,1e-5) =#

  #= #test2
  snap,_ = POD(full_snap,1e-5) =#

  #test3
  snap_tmp1,_ = POD(full_snap[:,1:Int(Q/2)],1e-5)
  snap_tmp2,_ = POD(full_snap[:,Int(Q/2)+1:end],1e-5)
  snap = hcat(snap_tmp1,snap_tmp2)

  return snap

end

function define_online_ST_stiffness(FEMSpace, RBInfo, el)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  Param = get_ParamInfo(problem_ntuple, RBInfo, μₒₙ)
  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2)
  A(t) = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α(t) * ∇(FEMSpace.ϕᵤ(t)))) * dΩ_sparse, FEMSpace.V(t), FEMSpace.V₀)

  for (nₜ,t) in enumerate(timesθ)
    println("Online snap, considering time step $nₜ/50")
    i,j,v = findnz(A(t))
    if nₜ === 1
      global Aₒₙ = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*length(timesθ))
    end
    global Aₒₙ[:,(nₜ-1)*FEMSpace.Nₛᵘ+1:nₜ*FEMSpace.Nₛᵘ] = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
  end

  return Aₒₙ

end

function define_online_ST_stiffness_old(FEMSpace, RBInfo, el, row_idx)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  Param = get_ParamInfo(problem_ntuple, RBInfo, μₒₙ)
  Acom(t) = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α(t) * ∇(FEMSpace.ϕᵤ(t)))) * FEMSpace.dΩ, FEMSpace.V(t), FEMSpace.V₀)

  for (nₜ,t) = enumerate(timesθ)
    println("Considering time step $nₜ/50")
    Aₒₙₜ_com = Matrix(reshape(Acom(t)[:][row_idx],:,1))
    if nₜ === 1
      global Aₒₙ_com = zeros(length(Aₒₙₜ_com)*length(timesθ),1)
    end
    global Aₒₙ_com[(nₜ-1)*length(Aₒₙₜ_com)+1:nₜ*length(Aₒₙₜ_com)] = Aₒₙₜ_com
  end

  return Aₒₙ_com

end

function define_online_ST_stiffness_fast(FEMSpace, RBInfo, el, MDEIM_idx_sparse)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  Param = get_ParamInfo(problem_ntuple, RBInfo, μₒₙ)
  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2)
  _, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse, FEMSpace.Nₛᵘ^2)
  idx_time = sort(idx_time)
  unique!(idx_time)
  A(t) = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α(t) * ∇(FEMSpace.ϕᵤ(t)))) * dΩ_sparse, FEMSpace.V(t), FEMSpace.V₀)

  for (nₜ,t) in enumerate(timesθ[idx_time])
    println("Online snap, considering time step $nₜ/$(length(idx_time))")
    i,j,v = findnz(A(t))
    if nₜ === 1
      global Aₒₙ = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*length(idx_time))
    end
    global Aₒₙ[:,(nₜ-1)*FEMSpace.Nₛᵘ+1:nₜ*FEMSpace.Nₛᵘ] = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
  end

  return Aₒₙ

end

function run_multiple_tests_ST_MDEIM(FEMSpace,RBInfo)

  err_vec = Float64[]
  time_vec = Float64[]
  Q_vec = Int64[]

  #err1,time1,Q1 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 1)
  err2,time2,Q2 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 3)
  err3,time3,Q3 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 6)
  err4,time4,Q4 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 9)
  err5,time5,Q5 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 12)
  err6,time6,Q6 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 15)
  err7,time7,Q7 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 18)
  err8,time8,Q8 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 21)
  err9,time9,Q9 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 24)
  err10,time10,Q10 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 27)
  err11,time11,Q11 = run_one_test_ST_MDEIM(FEMSpace, RBInfo, 30)

  err_vec = [err2,err3,err4,err5,err6,err7,err8,err9,err10,err11]
  time_vec = [time2,time3,time4,time5,time6,time7,time8,time9,time10,time11]
  Q_vec = [Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11]


  #= save_CSV(err_vec,"/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/ST-GRB_st/results/err_vec.csv")
  save_CSV(time_vec,"/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/ST-GRB_st/results/times_vec.csv")
 =#
  (err_vec,time_vec,Q_vec)

end

#FEMSpace = simple_model()
err_vec,time_vec,Q_vec = run_multiple_tests_ST_MDEIM(FEMSpace,RBInfo)


################################################################################
################################################################################
                              #OLD MDEIM CODE
################################################################################
################################################################################

function M_DEIM_POD_old(S::SparseMatrixCSC, ϵ = 1e-5)

  S̃ = copy(S)
  Nₕ = size(S̃)[1]
  row_idx, _, _ = findnz(S̃)
  unique!(row_idx)
  S̃ = Matrix(S̃[row_idx, :])

  M_DEIM_mat, Σ, _ = svd(S̃)

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0
  M_DEIM_err_bound = 1e5
  nₛ = size(S̃)[2]
  crit_val = norm(inv(M_DEIM_mat'M_DEIM_mat))
  mult_factor = sqrt(nₛ) * crit_val

  while N ≤ size(S̃)[2]-2 && (M_DEIM_err_bound > ϵ || cumulative_energy / total_energy < 1.0 - ϵ ^ 2)

    N += 1
    cumulative_energy += Σ[N] ^ 2
    M_DEIM_err_bound = Σ[N + 1] * mult_factor

    println("(M)DEIM-POD loop number $N, projection error = $M_DEIM_err_bound")

  end
  M_DEIM_mat = M_DEIM_mat[:, 1:N]
  _, col_idx, val = findnz(sparse(M_DEIM_mat))
  sparse_M_DEIM_mat = sparse(repeat(row_idx, N), col_idx, val, Nₕ, N)

  println("Basis number obtained via POD is $N, projection error ≤ $M_DEIM_err_bound")
  return sparse_M_DEIM_mat, Σ

end

function M_DEIM_offline_old(sparse_M_DEIM_mat::SparseMatrixCSC, Σ::Vector)

  row_idx, _, _ = findnz(sparse_M_DEIM_mat)
  unique!(row_idx)
  M_DEIM_mat = Matrix(sparse_M_DEIM_mat[row_idx, :])

  (N, n) = size(M_DEIM_mat)
  n_new = n
  M_DEIM_idx = Int64[]
  append!(M_DEIM_idx, convert(Int64, argmax(abs.(M_DEIM_mat[:, 1]))[1]))
  for m in range(2, n)
    res = M_DEIM_mat[:, m] - M_DEIM_mat[:, 1:(m-1)] * (M_DEIM_mat[M_DEIM_idx[1:(m-1)], 1:(m-1)] \ M_DEIM_mat[M_DEIM_idx[1:(m-1)], m])
    append!(M_DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      n_new = m
      break
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = Σ[min(n_new + 1,size(Σ)[1])] * norm(M_DEIM_mat[M_DEIM_idx,1:n_new] \ I(n_new))
  M_DEIM_idx = row_idx[M_DEIM_idx]

  sparse_M_DEIM_mat[:,1:n_new], M_DEIM_idx, M_DEIM_err_bound

end

function build_A_snapshots_old(FEMSpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  δtθ = RBInfo.δt*RBInfo.θ
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ

  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  A_t = assemble_stiffness(FEMSpace, RBInfo, Param)

  for i_t = 1:Nₜ
    println("Snapshot at time step $i_t, stiffness")
    A_i = A_t(timesθ[i_t])
    i, v = findnz(A_i[:])
    if i_t === 1
      global A = sparse(i, ones(length(i)), v, FEMSpace.Nₛᵘ^2, Nₜ)
    else
      global A[:, i_t] = sparse(i, ones(length(i)), v)
    end
  end

  A

end

function MDEIM_offline_old(FEMSpace::UnsteadyProblem, RBInfo::Info)

  println("Building $(RBInfo.nₛ_MDEIM) snapshots of $var, at each time step. This will take some time.")

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))

  for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k, need $(RBInfo.nₛ_MDEIM-k) more!")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))

    snapsₖ = build_A_snapshots_old(FEMSpace, RBInfo, μₖ)

    compressed_snapsₖ, _ = M_DEIM_POD_old(snapsₖ, RBInfo.ϵₛ)
    if k === 1
      global compressed_snaps_old = compressed_snapsₖ
    else
      global compressed_snaps_old = hcat(compressed_snaps_old, compressed_snapsₖ)
    end
  end

  sparse_MDEIM_mat, Σ = M_DEIM_POD_old(compressed_snaps_old, RBInfo.ϵₛ)
  MDEIM_mat_old, MDEIM_idx_old, MDEIM_err_bound = M_DEIM_offline_old(sparse_MDEIM_mat, Σ)

  Nₕ = convert(Int64, sqrt(size(MDEIM_mat_old)[1]))
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx_old, Nₕ)

  el_old = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(union(r_idx, c_idx)))

  MDEIM_mat_old, MDEIM_idx_old, el_old, MDEIM_err_bound, Σ

end

function assemble_MDEIM_matrices_old(RBInfo::Info, RBVars::PoissonSTGRB)

  MDEIM_mat_old, MDEIM_idx_old, sparse_el_old, _, _ = MDEIM_offline_old(FEMSpace, RBInfo)
  Q = size(MDEIM_mat_old)[2]
  #= Matₙ = zeros(RBVars.S.nₛᵘ, RBVars.S.nₛᵘ, Q)
  for q = 1:Q
    println("ST-GRB: affine component number $q, matrix $var"
    Matq = reshape(MDEIM_mat[:,q], (RBVars.S.Nₛᵘ, RBVars.S.Nₛᵘ))
    Matₙ[:,:,q] = RBVars.S.Φₛᵘ' * Matrix(Matq) * RBVars.S.Φₛᵘ
  end =#
  MDEIMᵢ_mat_old = Matrix(MDEIM_mat_old[MDEIM_idx_old, :])

  return MDEIM_mat_old, MDEIMᵢ_mat_old, MDEIM_idx_old, sparse_el_old

end

function get_θᵃ_old(RBVars,MDEIM_mat_old, MDEIM_idx_old, MDEIMᵢ_mat_old, sparse_el_old)

  Qold = size(MDEIM_mat_old)[2]
  A_μ_sparse_old = build_sparse_mat_old(FEMInfo, FEMSpace, RBInfo, sparse_el_old)
  Nₛᵘ = RBVars.S.Nₛᵘ
  θᵃ_old = zeros(RBVars.S.Qᵃ, RBVars.Nₜ)
  for iₜ = 1:RBVars.Nₜ
    θᵃ_old[:,iₜ] = M_DEIM_online(A_μ_sparse_old[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ], MDEIMᵢ_mat_old, MDEIM_idx_old)
  end

  θᵃ_old = reshape(θᵃ, Qold, RBVars.Nₜ)

  return θᵃ_old

end

function build_sparse_mat_old(FEMInfo::ProblemInfoUnsteady, FEMSpace::UnsteadyProblem, RBInfo::Info, el::Vector)

  μ=load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  μ_nb = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  Param = get_ParamInfo(problem_ntuple, RBInfo, μ_nb)

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  Nₜ = convert(Int64, RBInfo.T / RBInfo.δt)

  function define_Matₜ(t::Real)
    return assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α(t) * ∇(FEMSpace.ϕᵤ(t)))) * dΩ_sparse, FEMSpace.V(t), FEMSpace.V₀)
  end
  Matₜ(t) = define_Matₜ(t)

  i,j,v = findnz(Matₜ(timesθ[1]))
  Mat = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*Nₜ)
  for (i_t,t) in enumerate(timesθ[2:end])
    i,j,v = findnz(Matₜ(t))
    Mat[:,i_t*FEMSpace.Nₛᵘ+1:(i_t+1)*FEMSpace.Nₛᵘ] = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
  end

  Mat

end


function test_old_MDEIM(RBInfo, RBVars)
  MDEIM_mat_old, MDEIM_idx_old, sparse_el_old,_,_ = MDEIM_offline_old(FEMSpace, RBInfo)
  MDEIMᵢ_mat_old = MDEIM_mat_old[MDEIM_idx_old,:]
  μ=load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  μ_nb = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  #Param = get_ParamInfo(problem_ntuple, RBInfo, μ_nb)
  A = build_A_snapshots_old(FEMSpace, RBInfo, μ_nb)
  θᵃ_old = get_θᵃ_old(RBVars,MDEIM_mat_old, MDEIM_idx_old, MDEIMᵢ_mat_old, sparse_el_old)
  Aapprox = MDEIM_mat_old*θᵃ_old
  norm(A-Aapprox)
end
