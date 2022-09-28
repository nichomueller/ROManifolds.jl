function FE_solve(
  FEMSpace::FEMSpacePoissonS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  a(u, v) = ∫(∇(v) ⋅ (Param.α * ∇(u))) * FEMSpace.dΩ
  rhs(v) = ∫(v * Param.f) * FEMSpace.dΩ + ∫(v * Param.h) * FEMSpace.dΓn
  operator = AffineFEOperator(a, rhs, FEMSpace.V, FEMSpace.V₀)

  if FEMInfo.solver == "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  get_free_dof_values(uₕ_field)

end

function FE_solve(
  FEMSpace::FEMSpacePoissonST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  m(t, u, v) = ∫(Param.m(t)*(u*v)) * FEMSpace.dΩ
  a(t, u, v) = ∫(∇(v)⋅(Param.α(t)*∇(u)))*FEMSpace.dΩ
  rhs(t, v) = rhs_form(t,v,FEMSpace,Param)
  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.V, FEMSpace.V₀)

  linear_solver = LUSolver()

  if FEMInfo.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, FEMInfo.δt, FEMInfo.θ)
  else
    ode_solver = RungeKutta(linear_solver, FEMInfo.δt, FEMInfo.RK_type)
  end

  u₀_field = interpolate_everywhere(Param.u₀, FEMSpace.V(FEMInfo.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, FEMInfo.t₀, FEMInfo.tₗ)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, Int(FEMInfo.tₗ / FEMInfo.δt))
  global count = 0
  for (uₕ, _) in uₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(uₕ)::Vector{Float}
  end

  return uₕₜ

end

function FE_solve(
  FEMSpace::FEMSpaceADRS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS)

  a(u,v) = ∫(∇(v)⋅(Param.α*∇(u)) +
    v * Param.b ⋅ ∇(u) + Param.σ * v * u) * FEMSpace.dΩ
  lhs(u,v) = a(u,v)

  rhs(v) = ∫(v * Param.f) * FEMSpace.dΩ + ∫(v * Param.h) * FEMSpace.dΓn

  operator = AffineFEOperator(lhs, rhs, FEMSpace.V, FEMSpace.V₀)

  if FEMInfo.solver == "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  get_free_dof_values(uₕ_field)

end

function FE_solve(
  FEMSpace::FEMSpaceADRST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST)

  m(t, u, v) = ∫(Param.m(t)*(u*v)) * FEMSpace.dΩ
  a(t, u, v) = ∫(∇(v)⋅(Param.α(t)*∇(u)) +
    v * Param.b(t) ⋅ ∇(u) + Param.σ(t) * v * u) * FEMSpace.dΩ
  lhs(t,u,v) = a(t,u,v)
  rhs(t, v) = rhs_form(t,v,FEMSpace,Param)

  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.V, FEMSpace.V₀)

  linear_solver = LUSolver()

  if FEMInfo.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, FEMInfo.δt, FEMInfo.θ)
  else
    ode_solver = RungeKutta(linear_solver, FEMInfo.δt, FEMInfo.RK_type)
  end

  u₀_field = interpolate_everywhere(Param.u₀, FEMSpace.V(FEMInfo.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, FEMInfo.t₀, FEMInfo.tₗ)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, Int(FEMInfo.tₗ / FEMInfo.δt))
  global count = 0
  for (uₕ, _) in uₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(uₕ)::Vector{Float}
  end

  return uₕₜ

end

function FE_solve(
  FEMSpace::FEMSpaceStokesS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS) where T

  a((u,p),(v,q)) = ∫( ∇(v)⊙(Param.α*∇(u)) - Param.b*(∇⋅v)*p + q*(∇⋅u) ) * FEMSpace.dΩ
  rhs((v,q)) = ∫(v ⋅ Param.f) * FEMSpace.dΩ + ∫(v ⋅ Param.h) * FEMSpace.dΓn
  operator = AffineFEOperator(a, rhs, FEMSpace.X, FEMSpace.X₀)

  if FEMInfo.solver == "lu"
    uₕ_field, pₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field, pₕ_field = solve(LinearFESolver(), operator)
  end

  Vector(get_free_dof_values(uₕ_field)), Vector(get_free_dof_values(pₕ_field))

end

function FE_solve(
  FEMSpace::FEMSpaceStokesST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST) where T

  timesθ = get_timesθ(FEMInfo)
  θ = FEMInfo.θ
  δt = FEMInfo.δt
  δtθ = θ*δt

  A(t) = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "A")(t)
  M(t) = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "M")(t)
  B(t) = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "Bₚ")(t)
  F(t) = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "F")(t)
  H(t) = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "H")(t)
  R(t) = assemble_FEM_structure(FEMSpace, FEMInfo, Param, "L")(t)
  LHS(t) = vcat(hcat(M(t)/δtθ+A(t),-B(t)'),hcat(B(t),zeros(T,FEMSpace.Nₛᵖ,FEMSpace.Nₛᵖ)))
  RHS(t) = vcat(F(t)+H(t), zeros(T,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)) - R(t)

  u0(x) = Param.x₀(x)[1]
  p0(x) = Param.x₀(x)[2]
  u₀ = get_free_dof_values(interpolate_everywhere(u0, FEMSpace.V(FEMInfo.t₀)))::Vector{Float}
  p₀ = collect(get_free_dof_values(interpolate_everywhere(p0, FEMSpace.Q(FEMInfo.t₀))))::Vector{Float}

  uₕₜ = hcat(u₀, zeros(FEMSpace.Nₛᵘ, Int(FEMInfo.tₗ / FEMInfo.δt)))
  pₕₜ = hcat(p₀, zeros(FEMSpace.Nₛᵖ, Int(FEMInfo.tₗ / FEMInfo.δt)))

  count = 1
  for (iₜ,t) in enumerate(timesθ)
    count += 1
    xₕₜ = LHS(t)\(RHS(t)+vcat(M(t)/δtθ*uₕₜ[:,count-1], zeros(FEMSpace.Nₛᵖ)))
    uₕₜ[:,count] = 1/θ * xₕₜ[1:FEMSpace.Nₛᵘ] + (1-1/θ) * uₕₜ[:,count-1]
    pₕₜ[:,count] = 1/θ * xₕₜ[FEMSpace.Nₛᵘ+1:end] + (1-1/θ) * pₕₜ[:,count-1]
  end

  return uₕₜ[:,2:end], pₕₜ[:,2:end]

end

function FE_solve(
  FEMSpace::FEMSpaceNavierStokesS,
  ::FEMInfoS,
  Param::ParamInfoS) where T

  a((u,p),(v,q)) = ∫( ∇(v)⊙(Param.α*∇(u)) - Param.b*(∇⋅v)*p + q*(∇⋅u) ) * FEMSpace.dΩ

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) ) * FEMSpace.dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) ) * FEMSpace.dΩ

  rhs((v,q)) = ∫(v ⋅ Param.f) * FEMSpace.dΩ + ∫(v ⋅ Param.h) * FEMSpace.dΓn

  res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) - rhs((v,q))
  jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

  operator = FEOperator(res, jac, FEMSpace.X, FEMSpace.X₀)

  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
  solver = FESolver(nls)
  uₕ_field, pₕ_field = solve(solver, operator)

  Vector(get_free_dof_values(uₕ_field)), Vector(get_free_dof_values(pₕ_field))

end

#= function FE_solve(
  FEMSpace::FEMSpaceStokesST, probl::ProblemInfoUnsteady{T},
  Param::ParamInfoST{D,T}; subtract_Ddata=false)

  m(t,(u,p),(v,q)) = ∫(Param.m(t)*(u⋅v))*FEMSpace.dΩ
  ab(t,(u,p),(v,q)) = (∫(Param.α(t)*(∇(v) ⊙ ∇(u)))*FEMSpace.dΩ -
    ∫((∇⋅v)*p)*FEMSpace.dΩ + ∫(q*(∇⋅u))*FEMSpace.dΩ)
  rhs(t,(v,q)) = rhs_form(t,v,FEMSpace,Param)
  operator = TransientAffineFEOperator(m, ab, rhs, FEMSpace.X, FEMSpace.X₀)

  linear_solver = LUSolver()

  if probl.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u0(x) = Param.u₀(x)[1]
  p0(x) = Param.u₀(x)[2]
  u₀_field = interpolate_everywhere(u0, FEMSpace.V(probl.t₀))
  p₀_field = interpolate_everywhere(p0, FEMSpace.Q(probl.t₀))
  x₀_field = interpolate_everywhere([u₀_field,p₀_field], FEMSpace.X(probl.t₀))

  xₕₜ_field = solve(ode_solver, operator, x₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, convert(Int, probl.T / probl.δt))
  pₕₜ = zeros(FEMSpace.Nₛᵖ, convert(Int, probl.T / probl.δt))
  global count = 0
  for (xₕ, _) in xₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(xₕ[1])
    pₕₜ[:, count] = get_free_dof_values(xₕ[2])
  end

  if subtract_Ddata
    R₁,R₂ = assemble_lifting(FEMSpace, probl, Param)
    uₕₜ -= R₁
    pₕₜ -= R₂
  end

  return uₕₜ, pₕₜ

end =#

#= function FE_solve(FEMSpace::FEMSpacePoisson, Param::ParamInfoST{D,T}; subtract_Ddata = true)

_, Gₕ = get_lifting_operator(FEMSpace, Param)

res(u,v) = ∫( ∇(v) ⊙ (Param.α∘u ⋅ ∇(u)) - v * f) * FEMSpace.dΩ - ∫(v * Param.h) * FEMSpace.dΓn
jac(u, du, v) = ∫( ∇(v) ⊙ (Param.dα∘(du, ∇(u))) )*FEMSpace.dΩ
operator = FEOperator(res, jac, FEMSpace.V, FEMSpace.V₀)

nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

initial_uₕ_guess = FEFunction(FEMSpace.V, rand(Float, num_free_dofs(FEMSpace.V)))
uₕ_field, _ = solve!(initial_uₕ_guess, solver, operator)

if subtract_Ddata
  uₕ = get_free_dof_values(uₕ_field) - Gₕ
else
  uₕ = get_free_dof_values(uₕ_field)
end

return uₕ, Gₕ

end =#

function rhs_form(
  t::Real,
  v::FEBasis,
  FEMSpace::FEMSpacePoissonST,
  Param::ParamInfoST)

  if !isnothing(FEMSpace.dΓn)
    return ∫(v*Param.f(t))*FEMSpace.dΩ + ∫(v*Param.h(t))*FEMSpace.dΓn
  else
    return ∫(v*Param.f(t))*FEMSpace.dΩ
  end
end

function rhs_form(
  t::Real,
  v::FEBasis,
  FEMSpace::FEMSpaceStokesST,
  Param::ParamInfoST)

  if !isnothing(FEMSpace.dΓn)
    return ∫(v⋅Param.f(t))*FEMSpace.dΩ + ∫(v⋅Param.h(t))*FEMSpace.dΓn
  else
    return ∫(v⋅Param.f(t))*FEMSpace.dΩ
  end
end
