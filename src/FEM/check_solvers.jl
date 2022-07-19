function check_dataset(
  RBInfo::Info,
  RBVars::PoissonUnsteady{T},
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
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")(0.0)

  if RBInfo.case == 0
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")(0.0)
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")(0.0)
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  elseif RBInfo.case == 1
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")(0.0)
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  else
    A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")
    F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
    LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
    RHS1 = RBInfo.δt*RBInfo.θ*(F(t¹_θ)+H*Param.hₜ(t¹_θ))
    LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
    mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
    RHS2 = RBInfo.δt*RBInfo.θ*(F(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  end

  my_u1 = LHS1\RHS1
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
