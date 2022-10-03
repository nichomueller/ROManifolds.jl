function check_dataset(
  RBInfo::Info,
  nb::Int) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Param = get_ParamInfo(RBInfo, μ[nb])

  A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  u = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, nb]

  A \ (F + H - L) ≈ u

end

function check_dataset(
  RBInfo::Info,
  RBVars::PoissonST{T},
  nb::Int) where T

  RBVars.Nₜ = Int(RBInfo.tₗ / RBInfo.δt)
  get_snapshot_matrix(RBInfo, RBVars)

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)
  Param = get_ParamInfo(RBInfo, μ[nb])

  t¹_θ = RBInfo.t₀+RBInfo.δt*RBInfo.θ
  t²_θ = t¹_θ+RBInfo.δt

  u1 = RBVars.Sᵘ[:,(nb-1)*RBVars.Nₜ+1]
  u2 = RBVars.Sᵘ[:,(nb-1)*RBVars.Nₜ+2]
  M = assemble_FEM_structure(FEMSpace, RBInfo, Param, "M")(0.0)

  if RBInfo.case == 0
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")(0.0)
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")(0.0)
    H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")(0.0)
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  elseif RBInfo.case == 1
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")(0.0)
    H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")(0.0)
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  elseif RBInfo.case == 2
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
    H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")(0.0)
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F(t¹_θ)+H*Param.hₜ(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  else
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
    H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F(t¹_θ)+H(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F(t²_θ)+H(t²_θ))-mat*u1
  end

  my_u1 = LHS1\RHS1
  my_u2 = LHS2\RHS2

  u1≈my_u1 && u2≈my_u2

end

function check_stokes_solver()

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Param = get_ParamInfo(RBInfo, μ[nb])

  u = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, nb]
  p = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, nb]
  x = vcat(u, p)

  A = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "A")
  B = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "B")
  F = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")
  Lc = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "Lc")

  LHS = vcat(hcat(A, -B'), hcat(B, zeros(T, FEMSpace.Nₛᵖ, FEMSpace.Nₛᵖ)))
  RHS = vcat(F + H - L, - Lc)

  LHS * x - RHS

end

function check_MDEIM_stokesS()
  RBVars.DEIM_mat_L, RBVars.DEIM_idx_L, RBVars.DEIMᵢ_L, RBVars.sparse_el_L =
    DEIM_offline(RBInfo,"L")
  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Param = get_ParamInfo(RBInfo, μ[95])
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")
  θˡ = M_DEIM_online(L, RBVars.DEIMᵢ_L, RBVars.DEIM_idx_L)
  Lapp = RBVars.DEIM_mat_L * θˡ
  errL = abs.(Lapp - L)
  maximum(abs.(errL))
end

function check_navier_stokes_solver()

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Param = get_ParamInfo(RBInfo, μ[nb])

  u = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, nb]
  p = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, nb]
  x = vcat(u, p)

  ufun = FEFunction(FEMSpace.V, u)

  A = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "A")
  B = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "B")
  C = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "C")
  D = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "D")
  F = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")
  Lc = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "Lc")

  RHS = vcat(F + H - L, - Lc)
  #RHS = vcat(F + 0*H - L, - Lc)

  function J(x)
    xvec = get_free_dof_values(x)
    uvec = xvec[1:FEMSpace.Nₛᵘ]
    u = FEFunction(FEMSpace.V, uvec)

    vcat(hcat(A+C(u)+D(u), -B'), hcat(B, zeros(T, FEMSpace.Nₛᵖ, FEMSpace.Nₛᵖ)))
  end

  function res(x)
    xvec = get_free_dof_values(x)
    uvec = xvec[1:FEMSpace.Nₛᵘ]
    u = FEFunction(FEMSpace.V, uvec)

    LHS = vcat(hcat(A+C(u), -B'), hcat(B, zeros(T, FEMSpace.Nₛᵖ, FEMSpace.Nₛᵖ)))
    LHS * xvec - RHS
  end

  res(FEFunction(FEMSpace.X, x))

  x₀ = FEFunction(FEMSpace.X, zeros(FEMSpace.Nₛᵘ + FEMSpace.Nₛᵖ))
  x₁ = x₀ - J(x₀) \ res(x₀)

  û = RBVars.Φₛᵘ' * u
  Capp = sum([RBVars.MDEIM_C.Mat[:,q] * û[q]
    for q = 1:size(RBVars.MDEIM_C.Mat, 2)])
  _, vc = findnz(C(ufun)[:])
  maximum(abs.(vc - Capp))

end

function get_matrix_vector_nl_problem(operator::FEOperator)
  # src/Algebra/NLSolvers.jl
  x₀ = FEFunction(FEMSpace.X, zeros(FEMSpace.Nₛᵘ + FEMSpace.Nₛᵖ))
  u₀ = FEFunction(FEMSpace.V, zeros(FEMSpace.Nₛᵘ))

  RHS = vcat(F + 0*H - 0*L, - 0*Lc)

  J₀ = jacobian(operator, x₀)
  LHS₀ = vcat(hcat(A + C(u₀), -B'), hcat(B, zeros(T, FEMSpace.Nₛᵖ, FEMSpace.Nₛᵖ)))
  res₀(x) = LHS₀ * get_free_dof_values(x) - RHS
  result = get_free_dof_values(x₀) - J₀ \ res₀(x₀)
  x₁ = FEFunction(FEMSpace.X, result)
  u₁ = FEFunction(FEMSpace.V, result[1:FEMSpace.Nₛᵘ])

  x1 = FEFunction(FEMSpace.X, ones(FEMSpace.Nₛᵘ + FEMSpace.Nₛᵖ))
  u1 = FEFunction(FEMSpace.V, ones(FEMSpace.Nₛᵘ))
  J1 = jacobian(operator, x1)
  D(u) = assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
    (∇(u)'⋅FEMSpace.ϕᵤ) )*FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)
  LHS1 = vcat(hcat(A + C(u1) + D(u1), -B'), hcat(B, zeros(T, FEMSpace.Nₛᵖ, FEMSpace.Nₛᵖ)))
  J1 ≈ LHS1

end

function check_dataset(RBInfo, RBVars, i)

  μ = load_CSV(Array{Float}[], joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))
  Param = get_ParamInfo(RBInfo, μ[i])

  u1 = RBVars.Sᵘ[:, (i-1)*RBVars.Nₜ+1]
  u2 = RBVars.Sᵘ[:, (i-1)*RBVars.Nₜ+2]
  M = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "M.csv"))
  A = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "A.csv"))
  F = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "F.csv"))
  H = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "H.csv"))

  t¹_θ = RBInfo.t₀+RBInfo.δt*RBInfo.θ
  t²_θ = t¹_θ+RBInfo.δt

  LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t¹_θ,μ[i]))
  RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
  my_u1 = LHS1\RHS1

  LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ,μ[i]))
  mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ,μ[i]))-M
  RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1
  u2≈my_u2

end

using ForwardDiff
using LinearAlgebra
using Test
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using Gridap.ODEs.ODETools: ThetaMethodNonlinearOperator

function get_A_b(op,u0_field,Δtθ,ode_solver)
  odeop = get_algebraic_operator(op)
  ode_cache = allocate_cache(odeop)
  u0 = get_free_dof_values(u0_field)
  uf = copy(u0)
  vθ = similar(u0)
  nl_cache = nothing
  tθ = Δtθ
  ode_cache = update_cache!(ode_cache,odeop,tθ)
  nlop = ThetaMethodNonlinearOperator(odeop,tθ,Δtθ,u0,ode_cache,vθ)
  nl_cache = solve!(uf,ode_solver.nls,nlop,nl_cache)
  nl_cache.A, nl_cache.b
end

LHS_1 , RHS_1 = get_A_b(op_SUPG,u0_field,Δtθ,ode_solver)
