function get_JinvₙResₙ(
  RBInfo::ROMInfoS{3},
  RBVars::ROMMethodS{3,T},
  Params::Vector{ParamInfoS}) where T

  Cₙu, Dₙu = assemble_function_matricesₙ(RBInfo, RBVars, Params)
  RHSₙ = vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2])::Matrix{T}

  block2 = zeros(T, RBVars.nₛ[1], RBVars.nₛ[2])
  block3 = zeros(T, RBVars.nₛ[2], RBVars.nₛ[1])
  block4 = zeros(T, RBVars.nₛ[2], RBVars.nₛ[2])
  LHSₙ_lin = vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]), hcat(RBVars.LHSₙ[3], block4))

  LHSₙ_nonlin1 = vcat(hcat(Cₙu, block2), hcat(block3, block4))
  LHSₙ_nonlin2 = vcat(hcat(Cₙu + Dₙu, block2), hcat(block3, block4))

  Jₙ(u::Function) = LHSₙ_lin(u) + LHSₙ_nonlin2(u)
  resₙ(u::FEFunction, x̂::Matrix{T}) = (LHSₙ_lin(u) + LHSₙ_nonlin1(u)) * x̂ - RHSₙ

  JinvₙResₙ(u::FEFunction, x̂::Matrix{T}) = (Jₙ(u) \ resₙ(u, x̂))::Matrix{T}
  JinvₙResₙ::Function
end

function newton(
  FEMSpace::FOMS,
  RBInfo::ROMInfoS{3},
  RBVars::ROMMethodS{3,T},
  Params::Vector{ParamInfoS},
  ϵ=1e-9,
  max_k=10) where T

  JinvₙResₙ = get_JinvₙResₙ(RBInfo, RBVars, Params)

  x̂mat = zeros(T, sum(RBVars.nₛ), 1)
  δx̂ = 1. .+ x̂mat
  u = FEFunction(FEMSpace.V[1], zeros(T, RBVars.Nₛᵘ))
  k = 0
  err = norm(δx̂)

  while k < max_k && err ≥ ϵ
    δx̂ = JinvₙResₙ(u, x̂mat)
    x̂mat -= δx̂
    u = FEFunction(FEMSpace.V[1], RBVars.Φₛ[1] * x̂mat[1:RBVars.nₛ[1]])
    k += 1
    err = norm(δx̂)
    println("Iter: $k; ||δx̂||₂: $(norm(δx̂))")
  end

  println("Newton-Raphson ended with iter: $k; ||δx̂||₂: $(norm(δx̂))")
  x̂mat::Matrix{T}

end

function assemble_LHSₙ(
  RBInfo::ROMInfoS{3},
  RBVars::ROMMethodS{3,T},
  Params::Vector{ParamInfoS}) where T

  Matsₙ = assemble_matricesₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  LHSₙ = vcat(hcat(Matsₙ[1], Matsₙ[2]), hcat(Matsₙ[3],
    zeros(T, RBVars.nₛ[2], RBVars.nₛ[2])))
  push!(RBVars.LHSₙ, LHSₙ)

  return

end

function assemble_RHSₙ(
  RBInfo::ROMInfoS{3},
  RBVars::ROMMethodS{3,T},
  Params::Vector{ParamInfoS}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  RHSₙ = vcat(sum(Vecsₙ[1:3]), Vecsₙ[end])
  push!(RBVars.RHSₙ, RHSₙ)

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{3},
  RBVars::ROMMethodS{3,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  RHSₙ = vcat(RBVars.Φₛ[1]' * sum(RHS[1:3]), RBVars.Φₛ[2]' * RHS[end])
  push!(RBVars.RHSₙ, reshape(RHSₙ, :, 1)::Matrix{T})

  return

end

function solve_RB_system(
  FEMSpace::FOMS,
  RBInfo::ROMInfoS{3},
  RBVars::NavierStokesS{3,T},
  Params::Vector{ParamInfoS}) where T

  println("Solving RB problem via Newton-Raphson iterations")
  push!(RBVars.xₙ, newton(FEMSpace, RBInfo, RBVars, Params))

end

function assemble_solve_reconstruct(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{3},
  RBVars::ROM{3,T},
  μ::Vector{T}) where {D,T}
#careful -> need params
  assemble_RB_system(FEMSpace, RBInfo, RBVars, μ)
  RBVars.online_time += @elapsed begin
    solve_RB_system(RBVars)
  end
  reconstruct_FEM_solution(RBVars)

  return

end
