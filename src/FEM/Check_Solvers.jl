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

function ADR()
  #MODEL
  domain = (0,1,0,1)
  partition = (10,10)
  model = CartesianDiscreteModel(domain, partition)

  #DATA
  f(x,t::Real) = 1.
  f(t::Real) = x->f(x,t)
  g(x,t) = 0.
  g(t::Real) = x->g(x,t)
  b(x,t) = VectorValue(x[1],x[2])*t
  b(t::Real) = x->b(x,t)
  σ(x,t) = 1.
  σ(t::Real) = x->σ(x,t)
  b₂(x,t) = 0.5*b(x,t)
  b₂(t::Real) = x->b₂(x,t)
  div_b₂(t) = ∇⋅(b₂(t))
  div_b₂_σ(x,t::Real) = div_b₂(t)(x) + σ(x,t)
  div_b₂_σ(t::Real) = x->div_b₂_σ(x,t)

  #GRIDAP SOLVER
  order = 2
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary",conformity=:H1)
  U = TransientTrialFESpace(V,g)
  degree = 2*order
  Ωₕ = Triangulation(model)
  Qₕ = CellQuadrature(Ωₕ,4)
  dΩ = Measure(Ωₕ,degree)
  m(t,u,v) = ∫( v*u )dΩ
  a(t,u,v) = ∫( ∇(v)⋅∇(u) + v*(b(t) ⋅ ∇(u)) + σ(t)*v*u )dΩ
  rhs(t,v) = ∫(v*f(t))dΩ

  #SUPG STABILIZATION, SET ρ = 1 IF GLS STABILIZATION
  ρ = 0
  factor₁(t,u) = -Δ(u) + ∇⋅(b(t)*u) + σ(t)*u - f(t)
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,4)
  h = get_array(∫(1)dΛ)[1]
  pechlet(x,t::Real) = norm(b(x,t))*h / 2
  ξ(x) = coth(x) - 1/x
  τ(x,t::Real) = h*ξ(pechlet(x,t)) / (2*norm(b(x,t)))
  τ(t::Real) = x -> τ(x,t)
  lₛ(t,v) = - Δ(v) + div_bb₂_σ(t)*v
  lₛₛ(t,v) = bb(t) ⋅ ∇(v) + div_bb₂(t)*v
  factor₂(t,v) = τ(t) * (lₛ(t,v) + ρ*lₛₛ(t,v))
  l_stab(t,u,v) = ∫(factor₁(t,u)*factor₂(t,v)) * dΩ
  lhs(t,u,v) = a(t,u,v) + l_stab(t,u,v)

  ode_solver = ThetaMethod(LUSolver(), 0.01, 1)
  u0_field = interpolate_everywhere(0., U(0.))

  op_SUPG = TransientAffineFEOperator(m,lhs,rhs,U,V)
  uht_SUPG_form = solve(ode_solver, op_SUPG, u0_field, 0., 0.02)

  uht_SUPG = zeros(361, 2)
  global count = 0
  for (uh, _) in uht_SUPG_form
      global count += 1
      uht_SUPG[:, count] = get_free_dof_values(uh)
  end

  # MATRICES
  ϕᵥ = get_fe_basis(V)
  ϕᵤϕᵤ(t) = get_trial_fe_basis(U(t))

  M(t) = assemble_matrix(m(t,ϕᵤϕᵤ(t),ϕᵥ),U(t),V) / 0.01
  A(t) = assemble_matrix(a(t,ϕᵤϕᵤ(t),ϕᵥ),U(t),V)
  L(t) = assemble_matrix(l_stab(t,ϕᵤϕᵤ(t),ϕᵥ),U(t),V)
  F(t) = assemble_vector(rhs(t,ϕᵥ),V)
  LHS(t) = M(t)+A(t)+L(t)

  # THIS IS WRONG: RHS IS INCORRECT
  my_uht = zeros(361, 2)
  my_uht[:,1] = (M(0.01)+A(0.01)+L(0.01)) \ F(0.01)
  my_uht[:,2] = (M(0.02)+A(0.02)+L(0.02)) \ (F(0.02) + M(0.02)*my_uht[:,1])

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
