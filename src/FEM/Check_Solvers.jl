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

  using Gridap.FESpaces: residual_and_jacobian

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)));
  μ = μ[1]
  α = get_fun(FEMInfo, μ, "A")
  b = get_fun(FEMInfo, μ, "B")
  f = get_fun(FEMInfo, μ, "F")
  h = get_fun(FEMInfo, μ, "H")
  g = get_fun(FEMInfo, μ, "L")
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)
  FEMSpaceG = get_FEMSpace(FEMInfo, model, g)

  u = readdlm(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), ',', T)[:, 1]
  p = readdlm(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"), ',', T)[:, 1]
  x = vcat(u, p)

  A = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "A")
  B = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, "B")
  C = assemble_FEM_nonlinear_matrix(FEMSpaceG, FEMInfo, μ, "C")
  Cflat = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ, "C")
  D = assemble_FEM_nonlinear_matrix(FEMSpaceG, FEMInfo, μ, "D")
  F = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "F")
  H = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "H")
  L = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "L")
  Lc = assemble_FEM_vector(FEMSpace, FEMInfo, μ, "Lc")
  #= conv(u,∇u) = (∇u')⋅u
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )FEMSpaceG.dΩ
  ginterp = interpolate_dirichlet(g, FEMSpace.V[1])
  Lnew = assemble_vector(v->∫(α * ∇(v) ⊙ ∇(g))FEMSpace.dΩ + c(g,v),FEMSpace.V₀)
  Lcnew = assemble_vector(v->∫(b * v ⋅ (∇⋅(g)))FEMSpace.dΩ + c(g,v),FEMSpace.V) =#

  RHS = vcat(F + H + L, Lc)

  Nₛᵘ = length(get_free_dof_ids(FEMSpace.V₀[1]))
  Nₛᵖ = length(get_free_dof_ids(FEMSpace.V₀[2]))
  X₀ = MultiFieldFESpace(FEMSpaceG.V₀)
  X = MultiFieldFESpace(FEMSpaceG.V)
  Xflat = MultiFieldFESpace(FEMSpace.V)

  function J(x)
    u = x[1]
    vcat(hcat(A+C(u)+D(u), -B'), hcat(B, zeros(T, Nₛᵖ, Nₛᵖ)))
  end

  function res(xflat)
    xvecflat = get_free_dof_values(xflat)
    uflat = xflat[1]
    LHS = vcat(hcat(A+Cflat(uflat), -B'), hcat(B, zeros(T, Nₛᵖ, Nₛᵖ)))
    LHS * xvecflat - RHS
  end

  xfun = FEFunction(X, x)
  xfunflat = FEFunction(Xflat, x)
  rok, Jok = residual_and_jacobian(operator, x)

  a((u,p),(v,q)) = ∫( ∇(v)⊙(α*∇(u)) + b*(-(∇⋅v)*p + q*(∇⋅u)) )FEMSpaceG.dΩ
  rhs((v,q)) = ∫(v ⋅ f)FEMSpaceG.dΩ + ∫(v ⋅ h)FEMSpaceG.dΓn

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )FEMSpaceG.dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )FEMSpaceG.dΩ

  anew((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
  opAffine = AffineFEOperator(anew, rhs, X, X₀)
  LHSok = get_matrix(opAffine)
  RHSok = get_vector(opAffine)

  res(xfun)
  res(xfunflat)

  x₀ = FEFunction(X, zeros(Nₛᵘ + Nₛᵖ))
  x₁ = x₀ - J(x₀) \ res(x₀)

end

function check_dataset(RBInfo)

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  δtθ = RBInfo.δt*RBInfo.θ
  t¹_θ = RBInfo.t₀+δtθ

  u1 = readdlm(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), ',', T)[:, 1]
  p1 = readdlm(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"), ',', T)[:, 1]

  A(t) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ[1], "A", t)
  B(t) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ[1], "B", t)
  M(t) = assemble_FEM_matrix(FEMSpace, FEMInfo, μ[1], "M", t)
  Cfun = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ[1], "C")
  Dfun = assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, μ[1], "D")
  Vecs(t) = assemble_all_FEM_vectors(FEMSpace, FEMInfo, μ[1], t)

  Nₛᵘ = length(get_free_dof_ids(FEMSpace.V₀[1]))
  Nₛᵖ = length(get_free_dof_ids(FEMSpace.V₀[2]))

  L11(t,u) = A(t) + Cfun(u) + M(t)/δtθ
  J11(t,u) = A(t) + Cfun(u) + M(t)/δtθ + Dfun(u)
  L21(t) = B(t)
  LHS(t,u) = vcat(hcat(L11(t,u), -L21(t)'), hcat(L21(t), zeros(T, Nₛᵖ, Nₛᵖ)))
  RHS(t,u) = vcat(sum(Vecs(t)[1:3]) - M(t)*u/δtθ, Vecs(t)[end])

  function J(t,x)
    xvec = get_free_dof_values(x)
    uvec = xvec[1:Nₛᵘ]
    u = FEFunction(FEMSpace.V[1](t), uvec)

    vcat(hcat(J11(t,u), -L21(t)'), hcat(L21(t), zeros(T, Nₛᵖ, Nₛᵖ)))
  end

  function res(t,x,uprev)
    xvec = get_free_dof_values(x)
    uvec = xvec[1:Nₛᵘ]
    u = FEFunction(FEMSpace.V[1](t), uvec)

    LHS(t,u) * xvec - RHS(t,uprev)
  end

  x₀ = FEFunction(X(0.), zeros(Nₛᵘ + Nₛᵖ))
  my_x1θ = zeros(Nₛᵘ + Nₛᵖ) - J(t¹_θ,x₀) \ res(t¹_θ,x₀,zeros(Nₛᵘ))
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
