function check_dataset(
  RBInfo::Info,
  nb::Int) where T

  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)
  Param = ParamInfo(RBInfo, μ[nb])

  A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  u = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, nb]

  A \ (F + H - L) ≈ u

end

function check_dataset(RBInfo, RBVars, i)

  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)
  Param = ParamInfo(RBInfo, μ[i])

  u1 = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, (i-1)*RBVars.Nₜ+1]
  u2 = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, (i-1)*RBVars.Nₜ+2]
  A = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "A")
  M = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "M")(0.)
  F = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")

  δtθ = RBInfo.δt*RBInfo.θ
  t¹_θ = RBInfo.t₀+δtθ
  t²_θ = t¹_θ+RBInfo.δt

  RHS(t) = F(t)+H(t)-L(t)

  LHS1 = RBInfo.θ*(M/δtθ+A(t¹_θ))
  RHS1 = RHS(t¹_θ)
  my_u1 = LHS1\RHS1

  LHS2 = RBInfo.θ*(M/δtθ+A(t²_θ))
  mat = (1-RBInfo.θ)*A(t²_θ)-RBInfo.θ*M/δtθ
  RHS2 = RHS(t²_θ)-mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1
  u2≈my_u2

end

function check_dataset(RBInfo, RBVars, i)

  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)
  Param = ParamInfo(RBInfo, μ[i])

  u1 = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, (i-1)*RBVars.Nₜ+1]
  u2 = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, (i-1)*RBVars.Nₜ+2]
  p1 = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, (i-1)*RBVars.Nₜ+1]
  p2 = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, (i-1)*RBVars.Nₜ+2]

  A = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "A")
  B = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "B")(0.)
  M = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "M")(0.)
  F = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "F")(0.)
  H = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "H")(0.)
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")
  Lc = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "Lc")

  δtθ = RBInfo.δt*RBInfo.θ
  t¹_θ = RBInfo.t₀+δtθ
  t²_θ = t¹_θ+RBInfo.δt

  LHS(t) = vcat(hcat(M/δtθ+A(t), -B'), hcat(B, zeros(T, FEMSpace.Nₛᵖ, FEMSpace.Nₛᵖ)))
  RHS(t) = vcat(sin(t)*(F + H) - 0. *L(t), - 0. *Lc(t))

  my_x1θ = LHS(t¹_θ) \ RHS(t¹_θ)
  my_x1 = my_x1θ / RBInfo.θ
  my_u1 = my_x1[1:RBVars.Nₛᵘ]
  my_p1 = my_x1[RBVars.Nₛᵘ+1:end]

  my_x2θ = LHS(t²_θ) \ (RHS(t²_θ) + vcat(M/δtθ * my_u1, zeros(T, FEMSpace.Nₛᵖ)))
  my_x2 = (my_x2θ - (1-RBInfo.θ)*my_x1)/ RBInfo.θ
  my_u2 = my_x2[1:RBVars.Nₛᵘ]
  my_p2 = my_x2[RBVars.Nₛᵘ+1:end]

  u1≈my_u1
  u2≈my_u2
  p1≈my_p1
  p2≈my_p2

end

function check_stokes_solver()

  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)
  Param = ParamInfo(RBInfo, μ[nb])

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
  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)
  Param = ParamInfo(RBInfo, μ[95])
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")
  θˡ = MDEIM_online(L, RBVars.DEIMᵢ_L, RBVars.DEIM_idx_L)
  Lapp = RBVars.DEIM_mat_L * θˡ
  errL = abs.(Lapp - L)
  maximum(abs.(errL))
end

function check_navier_stokes_solver()

  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)
  Param = ParamInfo(RBInfo, μ[nb])

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

  û = RBVars.Φₛ' * u
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
