
mutable struct info
  Nₕ²::Int64; Nₜ::Int64; nₛ::Int64; ϵ::Float64;
end

function MDEIM_offline_spacetime(II::info,S::Matrix,row_idx) :: Tuple

  MDEIM_mat, Σ = POD(S, II.ϵ)
  MDEIM_mat, MDEIM_idx, _ = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(row_idx,MDEIM_idx,6640)
  MDEIM_idx_sparse_space, _ = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse,6640)
  el = find_FE_elements(FE_space.V₀, FE_space.Ω, unique(MDEIM_idx_sparse_space))

  return MDEIM_mat, MDEIM_idx, MDEIMᵢ_mat, el

end

function get_θₛₜ(II::info,MDEIM_idx,MDEIMᵢ_mat,online_snap) :: Array

  Q = length(MDEIM_idx)
  #MDEIM_idx_space, MDEIM_idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx, II.Nₕ²)
 #=  f = zeros(Q)
  for i = 1:Q
    ti = MDEIM_idx_time[i]
    f[i] = online_snap[(ti-1)*II.Nₕ²+1:ti*II.Nₕ²][MDEIM_idx_space[i]]
  end =#
  f = online_snap[MDEIM_idx]
  θ = MDEIMᵢ_mat\f

  return θ

end

function test_st_MDEIM(II::info,S::Matrix,row_idx)

  MDEIM_mat, MDEIM_idx, MDEIMᵢ_mat, el = MDEIM_offline_spacetime(II,S,row_idx)
  online_snap_com,online_snap_hyp = define_online_ST_stiffness(FE_space, ROM_info, el, row_idx)
  θ_com = get_θₛₜ(II,MDEIM_idx,MDEIMᵢ_mat,online_snap_com)
  θ_hyp = get_θₛₜ(II,MDEIM_idx,MDEIMᵢ_mat,online_snap_hyp)
  approx_snap_com = MDEIM_mat*θ_com
  approx_snap_hyp = MDEIM_mat*θ_hyp
  #online_snap_com,online_snap_hyp = Vector(reshape(online_snap_com,:,50)[row_idx,:][:]),Vector(reshape(online_snap_com,:,50)[row_idx,:][:])
  return reshape(online_snap_com,:,1), reshape(approx_snap_com,:,1), reshape(online_snap_hyp,:,1), reshape(approx_snap_hyp,:,1)

end

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
function run_one_test_ST_MDEIM(FE_space, ROM_info, nₛ_MDEIM::Int64)

  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  Nₜ = length(times_θ)
  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  μ = load_CSV(path)

  runtime = 0
  runtime += @elapsed begin
    for k = 1:nₛ_MDEIM
      @info "Considering parameter number $k, need $(nₛ_MDEIM-k) more!"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      param = get_parametric_specifics(ROM_info, μₖ)
      A_t = assemble_stiffness(FE_space, ROM_info, param)

      for i_t = 1:Nₜ
        @info "Snapshot at time step $i_t/Nₜ"
        A_i = A_t(times_θ[i_t])
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
  end

  runtime += @elapsed begin
    II = info(length(row_idx), Nₜ, nₛ_MDEIM, 1e-5)
    online_snap_com, approx_snap_com, online_snap_hyp, approx_snap_hyp = test_st_MDEIM(II,snaps,row_idx)
  end
  diff_com = online_snap_com - approx_snap_com
  err_com = [norm(diff_com)/norm(online_snap_com),sqrt(sum(diag(diff_com'*diff_com)))]
  diff_hyp = online_snap_hyp - approx_snap_hyp
  err_hyp = [norm(diff_hyp)/norm(online_snap_hyp),sqrt(sum(diag(diff_hyp'*diff_hyp)))]
  err_com,err_hyp,runtime
end

function define_online_ST_stiffness(FE_space, ROM_info, el, row_idx)
  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2)
  Acom(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
  Ahyp(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)

  for (nₜ,t) = enumerate(times_θ)
    @info "Considering time step $nₜ/50"
    Aₒₙₜ_com = Matrix(reshape(Acom(t)[:][row_idx],:,1))
    Aₒₙₜ_hyp = Matrix(reshape(Ahyp(t)[:][row_idx],:,1))
    if nₜ === 1
      global Aₒₙ_com = zeros(length(Aₒₙₜ_com)*length(times_θ),1)
      global Aₒₙ_hyp = zeros(length(Aₒₙₜ_hyp)*length(times_θ),1)
    else
      global Aₒₙ_com[(nₜ-1)*length(Aₒₙₜ_com)+1:nₜ*length(Aₒₙₜ_com)] = Aₒₙₜ_com
      global Aₒₙ_hyp[(nₜ-1)*length(Aₒₙₜ_com)+1:nₜ*length(Aₒₙₜ_com)] = Aₒₙₜ_hyp
    end
  end
  #= for (nₜ,t) in enumerate(times_θ)
    @info "Online snap, considering time step $nₜ/50"
    i,j,v = findnz(Acom(t))
    ihyp,jhyp,vhyp = findnz(Ahyp(t))
    if nₜ === 1
      global Aₒₙ_com = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*length(times_θ))
      global Aₒₙ_hyp = sparse(ihyp,jhyp,vhyp,FE_space.Nₛᵘ,FE_space.Nₛᵘ*length(times_θ))
    end
    global Aₒₙ_com[:,(nₜ-1)*FE_space.Nₛᵘ+1:nₜ*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
    global Aₒₙ_hyp[:,(nₜ-1)*FE_space.Nₛᵘ+1:nₜ*FE_space.Nₛᵘ] = sparse(ihyp,jhyp,vhyp,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
  end =#
  #vₒₙ = remove_zero_entries(Aₒₙ)[:]
  return Aₒₙ_com,Aₒₙ_hyp

end

function run_multiple_tests_ST_MDEIM(FE_space, ROM_info)

  err_vec = Float64[]
  time_vec = Float64[]
  nₛ_MDEIM = [20,40]

  err,err_hyp,rectime = run_one_test_ST_MDEIM(FE_space, ROM_info, 20)
  append!(err_vec,[err;err_hyp])
  append!(time_vec,rectime)


  #save_CSV(err_vec,"/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/ST-GRB_st/results/err_vec.csv")
  #save_CSV(time_vec,"/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/ST-GRB_st/results/times_vec.csv")

  return err_vec,time_vec

end

#err_vec,time_vec = run_multiple_tests_ST_MDEIM(FE_space, ROM_info)

################################################################################

#= function perform_test()

  right = Matrix{Float64}[]
  wrong = Matrix{Float64}[]
  right_eig = Matrix{Float64}[]
  wrong_eig = Matrix{Float64}[]
  norm_err = Float64[]
  rel_norm_err = Float64[]
  for Nₜ = 1:6:31
    II = info(100, Nₜ, 30, 1e-5)
    for nₜ = 1:II.Nₜ
      Sₜ = rand(II.Nₕ²,II.nₛ)
      if nₜ === 1
        global S = Sₜ
      else
        global S = vcat(S, Sₜ)
      end
    end
    online_snap = rand(II.Nₕ²*II.Nₜ)
    online_snap, approx_snap = test_st_MDEIM(II,S,online_snap)
    push!(right, online_snap)
    push!(wrong, approx_snap)
    append!(norm_err, norm(online_snap-approx_snap))
    append!(rel_norm_err, norm(online_snap-approx_snap)/norm(online_snap))

    if Nₜ > 1
      online_eig, approx_eig = check_eigen(online_snap,approx_snap,Nₜ)
      push!(right_eig, online_eig)
      push!(wrong_eig, approx_eig)
    end

  end

  right,wrong,right_eig,wrong_eig,norm_err,rel_norm_err

end

#perform_test() =#

#= """Steady case"""
function test_with_stiffness_standard_MDEIM(FE_space, ROM_info)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  μ = load_CSV(path)
  snaps = build_A_snapshots(FE_space, ROM_info, μ)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  Aₒₙ = reshape(assemble_stiffness(FE_space, ROM_info, param),:,1)
  time1 = @elapsed begin
    sparse_MDEIM_mat, sparse_Σ = M_DEIM_POD(snaps, ROM_info.ϵₛ)
    MDEIM_mat, MDEIM_idx, _ = M_DEIM_offline(sparse_MDEIM_mat, sparse_Σ)
    θ = Matrix(MDEIM_mat[MDEIM_idx,:])\Aₒₙ[MDEIM_idx,:]
    Aapp = MDEIM_mat*θ
    sparse_err = norm(Aapp-Aₒₙ)/norm(Aₒₙ)
  end

  time2 = @elapsed begin
    full_snaps = remove_zero_entries(snaps)
    _,_,vₒₙ = findnz(Aₒₙ)
    full_Aₒₙ = reshape(Vector(vₒₙ),:,1)
    MDEIM_mat, Σ = M_DEIM_POD(full_snaps, ROM_info.ϵₛ)
    MDEIM_mat, MDEIM_idx, _ = M_DEIM_offline(MDEIM_mat, Σ)
    full_θ = MDEIM_mat[MDEIM_idx,:]\full_Aₒₙ[MDEIM_idx,:]
    full_Aapp = MDEIM_mat*full_θ
    err = norm(full_Aapp-full_Aₒₙ)/norm(full_Aₒₙ)
  end

  sparse_err,err,time1,time2
end

"""Steady case"""
function test_with_stiffness_S_MDEIM(FE_space, ROM_info)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  μ = load_CSV(path)
  snaps = build_A_snapshots(FE_space, ROM_info, μ)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  Aₒₙ = reshape(assemble_stiffness(FE_space, ROM_info, param),:,1)

  time = @elapsed begin
    full_snaps = remove_zero_entries(snaps)
    _,_,vₒₙ = findnz(Aₒₙ)
    II = info(length(vₒₙ), 1, ROM_info.nₛ_MDEIM, 1e-5)
    online_snap, approx_snap = test_st_MDEIM(II,full_snaps,Vector(vₒₙ))
    err = norm(online_snap-approx_snap)/norm(online_snap)
  end

  err,time
end

"""Unsteady case"""
function test_with_stiffness_ST_MDEIM(FE_space, ROM_info)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  μ = load_CSV(path)
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  for (nₜ,t) = enumerate(times_θ)
    @info "Considering time step $nₜ/50"

    snapsₜ, row_idx = build_A_snapshots(FE_space, ROM_info, μ, t)

    if nₜ === 1
      global row_idx = row_idx
      global snaps = snapsₜ
    else
      global snaps = vcat(snaps, snapsₜ)
    end

  end

  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  for (nₜ,t) = enumerate(times_θ)
    @info "Considering time step $nₜ/50"
    Aₒₙₜ = reshape(assemble_stiffness(FE_space, ROM_info, param)(t),:,1)
    if nₜ === 1
      global Aₒₙ = Aₒₙₜ
    else
      global Aₒₙ = hcat(Aₒₙ, Aₒₙₜ)
    end
  end

  II = info(length(row_idx), Nₜ, ROM_info.nₛ_MDEIM, 1e-5)

  time = @elapsed begin
    full_Aₒₙ = remove_zero_entries(Aₒₙ)
    vₒₙ = full_Aₒₙ[:]
    online_snap, approx_snap = test_st_MDEIM(II,snaps,vₒₙ)
    err = norm(online_snap-approx_snap)/norm(online_snap)
  end

  err,time
end =#
