function check_dataset(
  RBInfo::Info,
  nb::Int) where T

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  Mats = assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ[1])
  Vecs = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ[1])

  u = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, nb]

  Mats[1] \ sum(Vecs) ≈ u

end

function check_stokes_solver()

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  u = readdlm(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), ',', T)[:,1]
  p = readdlm(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"), ',', T)[:,1]
  x = vcat(u, p)

  Mats = assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ[1])
  Vecs = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ[1])

  Nₛᵖ = length(get_free_dof_ids(FEMSpace.V₀[2]))

  LHS = vcat(hcat(Mats[1], -Mats[2]'), hcat(Mats[2], zeros(T, Nₛᵖ, Nₛᵖ)))
  RHS = vcat(sum(Vecs[1:3]), Vecs[4])

  LHS * x - RHS

end

function check_dataset()

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  δtθ = RBInfo.δt*RBInfo.θ
  t¹_θ = RBInfo.t₀+δtθ
  t²_θ = t¹_θ+RBInfo.δt

  Mats(t) = assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ[1], t)
  Vecs(t) = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ[1], t)

  u = readdlm(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), ',', T)
  u1, u2 = u[:, 1], u[:, 2]

  LHS1 = RBInfo.θ*(Mats(t¹_θ)[2]/δtθ+Mats(t¹_θ)[1])
  RHS1 = sum(Vecs(t¹_θ))
  my_u1 = LHS1\RHS1

  LHS2 = RBInfo.θ*(Mats(t²_θ)[2]/δtθ+Mats(t²_θ)[1])
  mat = (1-RBInfo.θ)*Mats(t²_θ)[1] - RBInfo.θ*Mats(t²_θ)[2]/δtθ
  RHS2 = sum(Vecs(t²_θ)) - mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1
  u2≈my_u2

end

function check_dataset(RBInfo)

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  δtθ = RBInfo.δt*RBInfo.θ
  t¹_θ = RBInfo.t₀+δtθ
  t²_θ = t¹_θ+RBInfo.δt

  u1 = readdlm(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), ',', T)[:, 1]
  u2 = readdlm(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), ',', T)[:, 2]
  p1 = readdlm(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"), ',', T)[:, 1]
  p2 = readdlm(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"), ',', T)[:, 2]

  Mats(t) = assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ[1], t)
  Vecs(t) = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ[1], t)

  Nₛᵘ = length(get_free_dof_ids(FEMSpace.V₀[1]))
  Nₛᵖ = length(get_free_dof_ids(FEMSpace.V₀[2]))

  L11(t) = Mats(t)[1] + Mats(t)[3]/δtθ
  L21(t) = Mats(t)[2]
  LHS(t) = vcat(hcat(L11(t), -L21(t)'), hcat(L21(t), zeros(T, Nₛᵖ, Nₛᵖ)))
  RHS(t) = vcat(sum(Vecs(t)[1:3]), Vecs(t)[end])

  my_x1θ = LHS(t¹_θ) \ RHS(t¹_θ)
  my_x1 = my_x1θ / RBInfo.θ
  my_u1 = my_x1[1:Nₛᵘ]
  my_p1 = my_x1[Nₛᵘ+1:end]

  my_x2θ = LHS(t²_θ) \ (RHS(t²_θ) + vcat(Mats(t²_θ)[3]/δtθ * my_u1, zeros(T, Nₛᵖ)))
  my_x2 = (my_x2θ - (1-RBInfo.θ)*my_x1)/ RBInfo.θ
  my_u2 = my_x2[1:Nₛᵘ]
  my_p2 = my_x2[Nₛᵘ+1:end]

  u1≈my_u1
  u2≈my_u2
  p1≈my_p1
  p2≈my_p2

end

function check_MDEIM_stokesS()
  RBVars.DEIM_mat_L, RBVars.DEIM_idx_L, RBVars.DEIMᵢ_L, RBVars.sparse_el_L =
    DEIM_offline(RBInfo,"L")
  FEMSpace, μ = get_FEMμ_info(RBInfo)
  Param = ParamInfo(RBInfo, μ[95])
  L = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")
  θˡ = MDEIM_online(L, RBVars.DEIMᵢ_L, RBVars.DEIM_idx_L)
  Lapp = RBVars.DEIM_mat_L * θˡ
  errL = abs.(Lapp - L)
  maximum(abs.(errL))
end

function check_navier_stokes_solver()

  FEMSpace, μ = get_FEMμ_info(RBInfo)
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

Q = 1
n = RBVars.nₛ .* RBVars.nₜ
Param = ParamInfo(RBInfo, μ[95], "B");
MDEIM = RBVars.Vars[2].MDEIM;
Mat = assemble_FEM_matrix(FEMSpace, RBInfo.FEMInfo, μ[95], "B", timesθ)';
Param.θ = θ(FEMSpace, RBInfo, Param, MDEIM);
Matn = zeros(n[1], n[2], Q);
function idxx1(i,j)
  (i-1)*RBVars.nₜ[1] + j
end
function idxx2(i,j)
  (i-1)*RBVars.nₜ[2] + j
end
for q = 1:Q
  for is = 1:RBVars.nₛ[1]
    for it = 1:RBVars.nₜ[1]
      i = idxx1(is,it)
      for js = 1:RBVars.nₛ[2]
        for jt = 1:RBVars.nₜ[2]
          j = idxx2(js,jt)
          Matn[i,j,q] = RBVars.Vars[2].Matₙ[q][js,is] * sum(RBVars.Φₜ[1][2:end,it] .* RBVars.Φₜ[2][1:end-1,jt] .* Param.θ[q][2:end])
        end
      end
    end
  end
end

Matred = sum([Matn[:,:,q] for q = 1:Q])
maximum(abs.(Matred - Mats₁ₙ[2]'))

#= Q = 1
n = RBVars.nₛ .* RBVars.nₜ
Param = ParamInfo(RBInfo, μ[95], "Lc");
MDEIM = RBVars.Vars[7].MDEIM;
Mat = assemble_FEM_vector(FEMSpace, RBInfo.FEMInfo, μ[95], "Lc", timesθ);
Param.θ = θ(FEMSpace, RBInfo, Param, MDEIM);
Matn = zeros(n[2], 1, Q);
function idxx1(i,j)
  (i-1)*RBVars.nₜ[1] + j
end
function idxx2(i,j)
  (i-1)*RBVars.nₜ[2] + j
end
for q = 1:Q
  for is = 1:RBVars.nₛ[2]
    for it = 1:RBVars.nₜ[2]
      i = idxx2(is,it)
      Matn[i,1,q] = RBVars.Vars[7].Matₙ[q][is,1] * sum(RBVars.Φₜ[1][:,it] .* Param.θ[q])
    end
  end
end

Matred = sum([Matn[:,:,q] for q = 1:Q])
maximum(abs.(Matred - Vecsₙ[4])) =#
