function FE_solve(
  FEMSpace::FEMSpacePoissonSteady,
  probl::ProblemInfoSteady{T},
  Param::ParametricInfoSteady{D,T};
  subtract_Ddata = true) where {D,T}

  Gₕ = assemble_lifting(FEMSpace, Param)

  a(u, v) = ∫(∇(v) ⋅ (Param.α * ∇(u))) * FEMSpace.dΩ
  rhs(v) = ∫(v * Param.f) * FEMSpace.dΩ + ∫(v * Param.h) * FEMSpace.dΓn
  operator = AffineFEOperator(a, rhs, FEMSpace.V, FEMSpace.V₀)

  if probl.solver == "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  if subtract_Ddata
    uₕ = get_free_dof_values(uₕ_field) - Gₕ
  else
    uₕ = get_free_dof_values(uₕ_field)
  end

  return uₕ, Gₕ

end

function FE_solve(
  FEMSpace::FEMSpacePoissonUnsteady,
  probl::ProblemInfoUnsteady{T},
  Param::ParametricInfoUnsteady{D,T};
  subtract_Ddata = true) where {D,T}

  Gₕₜ = assemble_lifting(FEMSpace, probl, Param)

  m(t, u, v) = ∫(Param.m(t)*(u*v))dΩ
  a(t, u, v) = ∫(∇(v)⋅(Param.α(t)*∇(u)))*FEMSpace.dΩ
  rhs(t, v) = rhs_form(t,v,FEMSpace,Param)
  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.V, FEMSpace.V₀)

  linear_solver = LUSolver()

  if probl.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u₀_field = interpolate_everywhere(Param.u₀, FEMSpace.V(probl.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, probl.t₀, probl.tₗ)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.tₗ / probl.δt))
  global count = 0
  dΩ = FEMSpace.dΩ
  for (uₕ, _) in uₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(uₕ)
  end

  if subtract_Ddata
    uₕₜ -= Gₕₜ
  end

  return uₕₜ, Gₕₜ

end

function FE_solve(
  FEMSpace::FEMSpaceStokesUnsteady,
  probl::ProblemInfoUnsteady{T},
  Param::ParametricInfoUnsteady{D,T};
  subtract_Ddata=false) where {D,T}

  timesθ = get_timesθ(probl)
  θ = probl.θ
  δt = probl.δt
  δtθ = θ*δt

  A(t) = assemble_stiffness(FEMSpace, FEMInfo, Param)(t)
  M(t) = assemble_mass(FEMSpace, FEMInfo, Param)(t)
  B(t) = assemble_primal_op(FEMSpace)(t)
  F(t) = assemble_forcing(FEMSpace, FEMInfo, Param)(t)
  H(t) = assemble_neumann_datum(FEMSpace, FEMInfo, Param)(t)
  R₁(t) = assemble_lifting(FEMSpace, probl, Param).R₁(t)
  R₂(t) = assemble_lifting(FEMSpace, probl, Param).R₂(t)
  Block1(t) = θ*(M(t)+δtθ*A(t))
  LHS(t) = vcat(θ*hcat(M(t)+δtθ*A(t),-δtθ*B(t)'),hcat(B(t),zeros(FEMSpace.Nₛᵖ,FEMSpace.Nₛᵖ)))
  RHS(t) = vcat(δtθ*(F(t)+H(t)-R₁(t)),-R₂(t))
  RHS_add(u,p,t) = θ*vcat((M(t)-δt*(1-θ)*A(t))*u+δt*(1-θ)*B(t)'*p,zeros(FEMSpace.Nₛᵖ))

  u0(x) = Param.u₀(x)[1]
  p0(x) = Param.u₀(x)[2]
  u₀ = get_free_dof_values(interpolate_everywhere(u0, FEMSpace.V(probl.t₀)))
  p₀ = get_free_dof_values(interpolate_everywhere(p0, FEMSpace.Q(probl.t₀)))

  uₕₜ = hcat(u₀,zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.tₗ / probl.δt)))
  pₕₜ = hcat(p₀,zeros(FEMSpace.Nₛᵖ, convert(Int64, probl.tₗ / probl.δt)))

  count = 1
  for (iₜ,t) in enumerate(timesθ)
    count += 1
    xₕₜ = LHS(t)\(RHS(t)+RHS_add(uₕₜ[:,count-1],pₕₜ[:,count-1],t-δt))
    uₕₜ[:,count] = xₕₜ[1:FEMSpace.Nₛᵘ]
    pₕₜ[:,count] = xₕₜ[FEMSpace.Nₛᵘ+1:end]
  end

  return uₕₜ[:,2:end], pₕₜ[:,2:end]

end

#= function FE_solve(
  FEMSpace::FEMSpaceStokesUnsteady, probl::ProblemInfoUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}; subtract_Ddata=false)

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
  uₕₜ = zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  pₕₜ = zeros(FEMSpace.Nₛᵖ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FEMSpace.dΩ
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

#= function FE_solve(FEMSpace::FEMSpacePoisson, Param::ParametricInfoUnsteady{D,T}; subtract_Ddata = true)

_, Gₕ = get_lifting_operator(FEMSpace, Param)

res(u,v) = ∫( ∇(v) ⊙ (Param.α∘u ⋅ ∇(u)) - v * f) * FEMSpace.dΩ - ∫(v * Param.h) * FEMSpace.dΓn
jac(u, du, v) = ∫( ∇(v) ⊙ (Param.dα∘(du, ∇(u))) )*dΩ
operator = FEOperator(res, jac, FEMSpace.V, FEMSpace.V₀)

nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

initial_uₕ_guess = FEFunction(FEMSpace.V, rand(Float64, num_free_dofs(FEMSpace.V)))
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
  FEMSpace::FEMSpacePoissonUnsteady,
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  if !isnothing(FEMSpace.dΓn)
    return ∫(v*Param.f(t))*FEMSpace.dΩ + ∫(v*Param.h(t))*FEMSpace.dΓn
  else
    return ∫(v*Param.f(t))*FEMSpace.dΩ
  end
end

function rhs_form(
  t::Real,
  v::FEBasis,
  FEMSpace::FEMSpaceStokesUnsteady,
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  if !isnothing(FEMSpace.dΓn)
    return ∫(v⋅Param.f(t))*FEMSpace.dΩ + ∫(v⋅Param.h(t))*FEMSpace.dΓn
  else
    return ∫(v⋅Param.f(t))*FEMSpace.dΩ
  end
end
