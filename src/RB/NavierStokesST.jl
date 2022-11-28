function get_JinvₙResₙ(
  RBInfo::ROMInfoST{3},
  RBVars::ROMMethodST{3,T},
  Params::Vector{ParamInfoST}) where T

  Cₙu, Dₙu = assemble_function_matricesₙ(RBInfo, RBVars, Params)
  RHSₙ = vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2])

  block2 = zeros(T, RBVars.nₛ[1], RBVars.nₛ[2])
  block3 = zeros(T, RBVars.nₛ[2], RBVars.nₛ[1])
  block4 = zeros(T, RBVars.nₛ[2], RBVars.nₛ[2])
  LHSₙ_lin = vcat(hcat(RBVars.LHSₙ[1], Matrix{T}(-RBVars.LHSₙ[2]')),
    hcat(RBVars.LHSₙ[2], zeros(T, RBVars.nₛ[2], RBVars.nₛ[2])))

  LHSₙ_nonlin1(u) = vcat(hcat(Cₙu(u), block2), hcat(block3, block4))
  LHSₙ_nonlin2(u) = vcat(hcat(Cₙu(u) + Dₙu(u), block2), hcat(block3, block4))

  Jₙ(u::FEFunction) = LHSₙ_lin + LHSₙ_nonlin2(u)
  resₙ(u::FEFunction, x̂::Matrix{T}) = (LHSₙ_lin + LHSₙ_nonlin1(u)) * x̂ - RHSₙ

  JinvₙResₙ(u::FEFunction, x̂::Matrix{T}) = (Jₙ(u) \ resₙ(u, x̂))::Matrix{T}
  JinvₙResₙ::Function
end

function newton(
  FEMSpace::FOMST{3,D},
  RBVars::ROMMethodST{3,T},
  JinvₙResₙ::Function,
  ϵ=1e-9,
  max_k=10) where {D,T}

  x̂mat = zeros(T, sum(RBVars.nₛ), 1)
  δx̂ = 1. .+ x̂mat
  u = FEFunction(FEMSpace.V[1], zeros(T, RBVars.Nₛ[1]))
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
  RBInfo::ROMInfoST{3},
  RBVars::ROMMethodST{3,T},
  Params::Vector{ParamInfoST}) where T

  Matsₙ, Mats₁ₙ = assemble_matricesₙ(RBInfo, RBVars, Params)
  LHSₙ11 = RBInfo.θ*(Matsₙ[1]+Matsₙ[3]) + (1-RBInfo.θ)*Mats₁ₙ[1] - RBInfo.θ*Mats₁ₙ[3]
  LHSₙ12 = - RBInfo.θ*Matsₙ[2]' - (1-RBInfo.θ)*Mats₁ₙ[2][2]
  LHSₙ21 = RBInfo.θ*Matsₙ[2] + (1-RBInfo.θ)*Mats₁ₙ[2][1]

  push!(RBVars.LHSₙ, LHSₙ11)
  push!(RBVars.LHSₙ, LHSₙ12)
  push!(RBVars.LHSₙ, LHSₙ21)

  return

end

function assemble_RHSₙ(
  RBInfo::ROMInfoST{3},
  RBVars::ROMMethodST{3,T},
  Params::Vector{ParamInfoST}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.RHSₙ, sum(Vecsₙ[1:3]))
  push!(RBVars.RHSₙ, Vecsₙ[end])

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMST{3,D},
  RBInfo::ROMInfoST{3},
  RBVars::ROMMethodST{3,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  RHS1 = blocks_to_matrix(getindex.(RHS,1) + getindex.(RHS,2) + getindex.(RHS,3))
  RHS2 = blocks_to_matrix(getindex.(RHS,4))
  push!(RBVars.RHSₙ, reshape((RHS1 * RBVars.Φₜ[1])' * RBVars.Φₛ[1], :, 1)::Matrix{T})
  push!(RBVars.RHSₙ, reshape((RHS2 * RBVars.Φₜ[2])' * RBVars.Φₛ[2], :, 1)::Matrix{T})

  return

end

function assemble_RB_system(
  FEMSpace::FOMST{3,D},
  RBInfo::ROMInfo{3},
  RBVars::ROM{3,T},
  μ::Vector{T}) where {D,T}

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = get_blocks_position(RBInfo)

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, blocks...)

    Params_lin = assemble_θ(FEMSpace, RBInfo, RBVars, μ)
    Params_nonlin = assemble_θ_function(FEMSpace, RBInfo, RBVars, μ)

    if "LHS" ∈ operators
      println("Assembling reduced LHS")
      assemble_LHSₙ(RBInfo, RBVars, Params_lin)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_rhs
        println("Assembling reduced RHS")
        assemble_RHSₙ(RBInfo, RBVars, Params_lin)
      else
        println("Assembling reduced RHS exactly")
        assemble_RHSₙ(FEMSpace, RBInfo, RBVars, μ)
      end
    end

    JinvₙResₙ = get_JinvₙResₙ(RBInfo, RBVars, Params_nonlin)
  end

  save_system_blocks(RBInfo, RBVars, operators, blocks...)

  JinvₙResₙ::Function

end

function solve_RB_system(
  FEMSpace::FOMST{3,D},
  RBVars::ROMMethodST{3,T},
  JinvₙResₙ::Function) where {D,T}

  println("Solving RB problem via Newton-Raphson iterations")
  xₙ = newton(FEMSpace, RBVars, JinvₙResₙ)
  n = RBVars.nₛ .* RBVars.nₜ
  push!(RBVars.xₙ, xₙ[1:n[1],:])
  push!(RBVars.xₙ, xₙ[n[1]+1:end,:])

end

function assemble_solve_reconstruct(
  RBInfo::ROMInfoST{3},
  RBVars::ROMMethodST{3,T},
  μ::Vector{T}) where T

  FEMSpace = get_FEMμ_info(RBInfo, μ, Val(get_FEM_D(RBInfo)))

  JinvₙResₙ = assemble_RB_system(FEMSpace, RBInfo, RBVars, μ)
  RBVars.online_time += @elapsed begin
    solve_RB_system(FEMSpace, RBVars, JinvₙResₙ)
  end
  reconstruct_FEM_solution(RBVars)

  return

end
