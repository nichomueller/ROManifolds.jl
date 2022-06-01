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

  refFE = ReferenceFE(lagrangian, Float64, 1)
  V₀ = TestFESpace(model, refFE; conformity=:H1)
  g₀(x, t::Real) = 0
  g₀(t::Real) = x->g₀(x,t)
  V = TransientTrialFESpace(V₀, g₀)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  FE_space = FESpacePoissonUnsteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FE_space

end

function MDEIM_offline_spacetime(II::info,S::Matrix,row_idx) :: Tuple

  MDEIM_mat, Σ = POD(S, II.ϵ)
  MDEIM_mat, MDEIM_idx, _ = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(row_idx,MDEIM_idx,FE_space.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse,FE_space.Nₛᵘ)
  el = find_FE_elements(FE_space.V₀, FE_space.Ω, unique(MDEIM_idx_sparse_space))

  return MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, el

end

function get_θₛₜ(MDEIM_idx,MDEIMᵢ_mat,online_snap) :: Array

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

function modify_MDEIM_idx(MDEIM_idx) :: Array

  idx_space, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx,FE_space.Nₛᵘ^2)
  idx_time_new = assign_label_to_sorted_elems(idx_time)
  MDEIM_idx_new = (idx_time_new.-1)*FE_space.Nₛᵘ^2+idx_space

  return MDEIM_idx_new

end

function test_st_MDEIM(FE_space,II::info,S::Matrix,row_idx)

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, el = MDEIM_offline_spacetime(II,S,row_idx)
  #online_snap = define_online_ST_stiffness(FE_space, ROM_info, el)
  online_snap_fast = define_online_ST_stiffness_fast(FE_space, ROM_info, el, MDEIM_idx_sparse)
  MDEIM_idx_new = modify_MDEIM_idx(MDEIM_idx_sparse)
  #θ = get_θₛₜ(MDEIM_idx_sparse,MDEIMᵢ_mat,online_snap)
  θ_fast = get_θₛₜ(MDEIM_idx_new,MDEIMᵢ_mat,online_snap_fast)
  #@test θ ≈ θfast
  approx_snap = MDEIM_mat*θ_fast

  return reshape(approx_snap,:,1), size(MDEIM_mat)[2], el

end

#= function check_similarity_approx_matrices()
  approx_snap = MDEIM_mat*θfast
  _, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse, FE_space.Nₛᵘ^2)
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
function run_one_test_ST_MDEIM(FE_space, ROM_info, nₛMDEIM::Int64)

  snaps, row_idx = define_offline_snaps(FE_space, ROM_info, nₛMDEIM)
  #snaps_red = define_offline_snaps_red(snaps)
  II = info(length(row_idx), 50, nₛMDEIM, 1e-5)

  runtime = @elapsed begin
    approx_snap, Q, el = test_st_MDEIM(FE_space,II,snaps,row_idx)
  end
  #= runtime_red = @elapsed begin
    approx_snap_red, Q_red = test_st_MDEIM(FE_space,II,snaps_red,row_idx)
  end =#
  online_snap = define_online_ST_stiffness_old(FE_space, ROM_info, el, row_idx)
  diff = online_snap - approx_snap
  #diff_red = online_snap - approx_snap_red
  #err = [norm(diff)/norm(online_snap),norm(diff_red)/norm(online_snap)]
  err = norm(diff)/norm(online_snap)
  err,runtime,Q
end

function define_offline_snaps(FE_space, ROM_info, nₛMDEIM::Int64)

  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  Nₜ = length(times_θ)
  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  μ = load_CSV(path)

  for k = 1:nₛMDEIM
    @info "Considering parameter number $k, need $(nₛMDEIM-k) more!"
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

function define_online_ST_stiffness(FE_space, ROM_info, el)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2)
  A(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)

  for (nₜ,t) in enumerate(times_θ)
    @info "Online snap, considering time step $nₜ/50"
    i,j,v = findnz(A(t))
    if nₜ === 1
      global Aₒₙ = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*length(times_θ))
    end
    global Aₒₙ[:,(nₜ-1)*FE_space.Nₛᵘ+1:nₜ*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
  end

  return Aₒₙ

end

function define_online_ST_stiffness_old(FE_space, ROM_info, el, row_idx)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  Acom(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)

  for (nₜ,t) = enumerate(times_θ)
    @info "Considering time step $nₜ/50"
    Aₒₙₜ_com = Matrix(reshape(Acom(t)[:][row_idx],:,1))
    if nₜ === 1
      global Aₒₙ_com = zeros(length(Aₒₙₜ_com)*length(times_θ),1)
    end
    global Aₒₙ_com[(nₜ-1)*length(Aₒₙₜ_com)+1:nₜ*length(Aₒₙₜ_com)] = Aₒₙₜ_com
  end

  return Aₒₙ_com

end

function define_online_ST_stiffness_fast(FE_space, ROM_info, el, MDEIM_idx_sparse)

  path = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/FEM_data/snapshots/μ.csv"
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  μ = load_CSV(path)
  μₒₙ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μₒₙ)
  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2)
  _, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx_sparse, FE_space.Nₛᵘ^2)
  idx_time = sort(idx_time)
  unique!(idx_time)
  A(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)

  for (nₜ,t) in enumerate(times_θ[idx_time])
    @info "Online snap, considering time step $nₜ/$(length(idx_time))"
    i,j,v = findnz(A(t))
    if nₜ === 1
      global Aₒₙ = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*length(idx_time))
    end
    global Aₒₙ[:,(nₜ-1)*FE_space.Nₛᵘ+1:nₜ*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
  end

  return Aₒₙ

end

function run_multiple_tests_ST_MDEIM(FE_space,ROM_info)

  err_vec = Float64[]
  time_vec = Float64[]
  Q_vec = Int64[]

  #err1,time1,Q1 = run_one_test_ST_MDEIM(FE_space, ROM_info, 1)
  err2,time2,Q2 = run_one_test_ST_MDEIM(FE_space, ROM_info, 3)
  err3,time3,Q3 = run_one_test_ST_MDEIM(FE_space, ROM_info, 6)
  err4,time4,Q4 = run_one_test_ST_MDEIM(FE_space, ROM_info, 9)
  err5,time5,Q5 = run_one_test_ST_MDEIM(FE_space, ROM_info, 12)
  err6,time6,Q6 = run_one_test_ST_MDEIM(FE_space, ROM_info, 15)
  err7,time7,Q7 = run_one_test_ST_MDEIM(FE_space, ROM_info, 18)
  err8,time8,Q8 = run_one_test_ST_MDEIM(FE_space, ROM_info, 21)
  err9,time9,Q9 = run_one_test_ST_MDEIM(FE_space, ROM_info, 24)
  err10,time10,Q10 = run_one_test_ST_MDEIM(FE_space, ROM_info, 27)
  err11,time11,Q11 = run_one_test_ST_MDEIM(FE_space, ROM_info, 30)

  err_vec = [err2,err3,err4,err5,err6,err7,err8,err9,err10,err11]
  time_vec = [time2,time3,time4,time5,time6,time7,time8,time9,time10,time11]
  Q_vec = [Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11]


  #= save_CSV(err_vec,"/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/ST-GRB_st/results/err_vec.csv")
  save_CSV(time_vec,"/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/3D_1/model.json/ST-GRB_st/results/times_vec.csv")
 =#
  (err_vec,time_vec,Q_vec)

end

#FE_space = simple_model()
err_vec,time_vec,Q_vec = run_multiple_tests_ST_MDEIM(FE_space,ROM_info)
