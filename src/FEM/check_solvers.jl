function check_dataset(
  FEMSpace::FEMProblem,
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  nb::Int64)

  μ = load_CSV(Array{Float64}[],joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  Param = get_ParamInfo(RBInfo, problem_id, μ[nb])

  u1 = RBVars.S.Sᵘ[:,(nb-1)*RBVars.Nₜ+1]
  u2 = RBVars.S.Sᵘ[:,(nb-1)*RBVars.Nₜ+2]
  M = assemble_FEM_structure(FEMSpace, RBInfo, Param, "M")(0.0)
  # we suppose that case == 1 --> no need to multiply A by αₜ
  A(t) = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")(t)
  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")(0.0)
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")(0.0)

  t¹_θ = RBInfo.t₀+RBInfo.δt*RBInfo.θ
  t²_θ = t¹_θ+RBInfo.δt

  LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
  RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
  my_u1 = LHS1\RHS1

  LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
  mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
  RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1 && u2≈my_u2

end

function check_stokes_solver()
  A = assemble_stiffness(FEMSpace, FEMInfo, Param)(0.0)
  M = assemble_mass(FEMSpace, FEMInfo, Param)(0.0)
  B = assemble_primal_op(FEMSpace)(0.0)
  F = assemble_forcing(FEMSpace, FEMInfo, Param)(0.0)
  H = assemble_neumann_datum(FEMSpace, FEMInfo, Param)(0.0)

  δt = 0.005
  θ = 0.5

  u1 = uₕ[:,1]
  p1 = pₕ[:,1]
  α = Param.α(0)(Point(0.,0.,0.))
  res1 = θ*(δt*θ*A*α + M)*u1 + δt*θ*B'*p1 - δt*θ*(F+H)
  res2 = B*u1

  u2 = uₕₜ[:,1]
  p2 = pₕₜ[:,1]
  res1 = θ*(δt*θ*A*α + M)*u2 + ((1-θ)*δt*θ*A*α - θ*M)*u1 + δt*θ*Bᵀ*p2 - δt*θ*(F+H)
  res2 = B*u2

end
