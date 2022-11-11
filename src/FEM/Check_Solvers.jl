function check_dataset(
  RBInfo::Info,
  nb::Int) where T

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  Mats = assemble_all_FEM_matrices(FEMSpace, FEMInfo, μ[1])
  Vecs = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ[1])

  u = Matrix{T}(CSV.read(joinpath(get_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, nb]

  Mats[1] \ sum(Vecs) ≈ u

end

function check_stokes_solver()

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  u = readdlm(joinpath(get_snap_path(RBInfo), "uₕ.csv"), ',', T)[:,1]
  p = readdlm(joinpath(get_snap_path(RBInfo), "pₕ.csv"), ',', T)[:,1]
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

  u = readdlm(joinpath(get_snap_path(RBInfo), "uₕ.csv"), ',', T)
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

  u1 = readdlm(joinpath(get_snap_path(RBInfo), "uₕ.csv"), ',', T)[:, 1]
  u2 = readdlm(joinpath(get_snap_path(RBInfo), "uₕ.csv"), ',', T)[:, 2]
  p1 = readdlm(joinpath(get_snap_path(RBInfo), "pₕ.csv"), ',', T)[:, 1]
  p2 = readdlm(joinpath(get_snap_path(RBInfo), "pₕ.csv"), ',', T)[:, 2]

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

  using Gridap.FESpaces: residual_and_jacobian

  μ = get_μ(RBInfo)
  μ = μ[95]
  FEMSpace = get_FEMμ_info(RBInfo, μ, Val(get_FEM_D(RBInfo)))

  X = MultiFieldFESpace(FEMSpace.V)
  u = readdlm(joinpath(get_snap_path(RBInfo), "uₕ.csv"), ',', Float)[:, 95]
  p = readdlm(joinpath(get_snap_path(RBInfo), "pₕ.csv"), ',', Float)[:, 95]
  x = vcat(u, p)
  xfun = FEFunction(X, x)
  Nₛᵖ = length(get_free_dof_ids(FEMSpace.V₀[2]))

  A = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "A")
  B = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "B")
  C = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, "C")
  D = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, "D")
  F = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "F")
  H = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "H")
  LA = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "LA")
  LB = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "LB")
  LC = assemble_FEM_nonlinear_vector(FEMSpace, FEMInfo, μ, "LC")

  LHS(u) = vcat(hcat(A+C(u), -B'), hcat(B, zeros(Nₛᵖ, Nₛᵖ)))
  J(xfun) = vcat(hcat(A+C(xfun[1])+D(xfun[1]), -B'), hcat(B, zeros(Nₛᵖ, Nₛᵖ)))
  RHS(u) = vcat(F + H + LA + LC(u), LB)
  res(xfun, xh) = LHS(xfun[1]) * xh - RHS(xfun[1])

  #LHS(xfun[1]) * x - RHS(xfun[1])

  function newton(res::Function, J::Function, x)
    err = 1.
    tolerance = 10^(-10)
    xh = get_free_dof_values(x)
    xn = Vector{Float}[]
    iter = 0
    while (norm(err) > tolerance)
      Jx, rx = J(x), res(x,xh)
      err = Jx \ rx
      xh -= err
      push!(xn,xh)
      x = FEFunction(X, xh)
      iter += 1
      println("err = $(norm(err)), iter = $iter")
    end
    xn
  end

  x₀ = zeros(length(x))
  xnew = newton(res, J, FEFunction(X, x₀))

end

function check_dataset(RBInfo)

  μ = get_μ(RBInfo)
  μ = μ[1]
  FEMSpace = get_FEMμ_info(RBInfo, μ, Val(get_FEM_D(RBInfo)))
  X = TransientMultiFieldFESpace(FEMSpace.V)

  δtθ = RBInfo.δt*RBInfo.θ
  t¹_θ = RBInfo.t₀+δtθ

  u1 = readdlm(joinpath(get_snap_path(RBInfo), "uₕ.csv"), ',')[:, 1]
  p1 = readdlm(joinpath(get_snap_path(RBInfo), "pₕ.csv"), ',')[:, 1]
  x1 = vcat(u1,p1)
  xfun1 = FEFunction(X(RBInfo.δt),x1)

  A(t) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "A", t)
  B(t) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "B", t)
  M(t) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "M", t)
  Cfun = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, "C")
  Dfun = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, "D")
  F(t) = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "F", t)
  H(t) = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "H", t)
  LA(t) = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "LA", t)
  LB(t) = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "LB", t)
  LC = assemble_FEM_nonlinear_vector(FEMSpace, FEMInfo, μ, "LC")

  Nₛᵘ = length(get_free_dof_ids(FEMSpace.V₀[1]))
  Nₛᵖ = length(get_free_dof_ids(FEMSpace.V₀[2]))

  L11(t,u) = A(t) + Cfun(u) + M(t)/δtθ
  J11(t,u) = A(t) + Cfun(u) + Dfun(u) + M(t)/δtθ
  L21(t) = B(t)
  LHS(t,u) = vcat(hcat(L11(t,u), -L21(t)'), hcat(L21(t), zeros(T, Nₛᵖ, Nₛᵖ)))
  RHS(t,u,uprev) = vcat(F(t) + H(t) + LA(t) + LC(u) - M(t)*uprev/δtθ, LB(t))

  J(t,x) = vcat(hcat(J11(t,x[1]), -L21(t)'), hcat(L21(t), zeros(T, Nₛᵖ, Nₛᵖ)))
  res(t,x,xh,xprev) = LHS(t,x[1]) * xh - RHS(t,x[1],xprev[1:Nₛᵘ])

  x₀h = zeros(Nₛᵘ + Nₛᵖ)
  x₀ = FEFunction(X(0.), x₀h)

  function newton(res::Function, J::Function, x, t)
    err = 1.
    tolerance = 10^(-10)
    xh = get_free_dof_values(x)
    iter = 0
    while (norm(err) > tolerance)
       err = J(t,x)\res(t,x,xh,x₀h)
       xh -= err
       x = FEFunction(X(t), xh[:,1])
       iter += 1
       println("err = $(norm(err)), iter = $iter")
    end
    xh
  end

  newtonδt(res::Function, J::Function, x) = newton(res, J, x, t¹_θ)
  my_x1θ = newtonδt(res, J, x₀)
  my_x1 = my_x1θ / RBInfo.θ
  my_u1 = my_x1[1:Nₛᵘ]
  my_p1 = my_x1[Nₛᵘ+1:end]


  u1≈my_u1
  p1≈my_p1

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
