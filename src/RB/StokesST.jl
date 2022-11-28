function assemble_LHSₙ(
  RBInfo::ROMInfoST{2},
  RBVars::ROMMethodST{2,T},
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
  RBInfo::ROMInfoST{2},
  RBVars::ROMMethodST{2,T},
  Params::Vector{ParamInfoST}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.RHSₙ, sum(Vecsₙ[1:3]))
  push!(RBVars.RHSₙ, Vecsₙ[end])

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMST{2,D},
  RBInfo::ROMInfoST{2},
  RBVars::ROMMethodST{2,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  RHS1 = blocks_to_matrix(getindex.(RHS,1) + getindex.(RHS,2) + getindex.(RHS,3))
  RHS2 = blocks_to_matrix(getindex.(RHS,4))
  push!(RBVars.RHSₙ, reshape((RHS1 * RBVars.Φₜ[1])' * RBVars.Φₛ[1], :, 1)::Matrix{T})
  push!(RBVars.RHSₙ, reshape((RHS2 * RBVars.Φₜ[2])' * RBVars.Φₛ[2], :, 1)::Matrix{T})

  return

end

function assemble_RB_system(
  FEMSpace::FOMST{2,D},
  RBInfo::ROMInfoST{2},
  RBVars::ROMMethodST{2,T},
  μ::Vector{T}) where {D,T}

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = get_blocks_position(RBInfo)

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, blocks...)

    Params = assemble_θ(FEMSpace, RBInfo, RBVars, μ)

    if "LHS" ∈ operators
      println("Assembling reduced LHS")
      assemble_LHSₙ(RBInfo, RBVars, Params)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_rhs
        println("Assembling reduced RHS")
        assemble_RHSₙ(RBInfo, RBVars, Params)
      else
        println("Assembling reduced RHS exactly")
        assemble_RHSₙ(FEMSpace, RBInfo, RBVars, μ)
      end
    end
  end

  save_system_blocks(RBInfo, RBVars, operators, blocks...)

  return

end

function solve_RB_system(RBVars::ROMMethodST{2,T}) where T

  println("Solving RB problem via backslash")
  n = RBVars.nₛ .* RBVars.nₜ
  LHSₙ = vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
    hcat(RBVars.LHSₙ[3], zeros(T, n[2], n[2])))
  RHSₙ = vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2])

  xₙ = LHSₙ \ RHSₙ
  push!(RBVars.xₙ, xₙ[1:n[1],:])
  push!(RBVars.xₙ, xₙ[n[1]+1:end,:])

  return

end

function assemble_solve_reconstruct(
  RBInfo::ROMInfo{2},
  RBVars::ROM{2,T},
  μ::Vector{T}) where T

  FEMSpace = get_FEMμ_info(RBInfo, μ, Val(get_FEM_D(RBInfo)))

  assemble_RB_system(FEMSpace, RBInfo, RBVars, μ)
  RBVars.online_time += @elapsed begin
    solve_RB_system(RBVars)
  end
  reconstruct_FEM_solution(RBVars)

  return

end
