function assemble_LHSₙ(
  RBInfo::ROMInfoST{1},
  RBVars::ROMMethodST{1,T},
  Params::Vector{ParamInfoST}) where T

  Matsₙ, Mats₁ₙ = assemble_matricesₙ(RBInfo, RBVars, Params)
  LHSₙ = RBInfo.θ*(sum(Matsₙ)) + (1-RBInfo.θ)*Mats₁ₙ[1] - RBInfo.θ*Mats₁ₙ[2]
  push!(RBVars.LHSₙ, LHSₙ)

  return

end

function assemble_RHSₙ(
  RBInfo::ROMInfoST{1},
  RBVars::ROMMethodST{1,T},
  Params::Vector{ParamInfoST}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.RHSₙ, sum(Vecsₙ))

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMST{1,D},
  RBInfo::ROMInfoST{1},
  RBVars::ROMMethodST{1,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  push!(RBVars.RHSₙ, reshape(RBVars.Φₛ[1]' * sum(RHS) *RBVars.Φₜ[1], :, 1)::Matrix{T})

  return

end

function assemble_RB_system(
  FEMSpace::FOMST{1,D},
  RBInfo::ROMInfoST{1},
  RBVars::ROMMethodST{1,T},
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
      if !RBInfo.online_RHS
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

function solve_RB_system(RBVars::ROMMethodST{1,T}) where T

  println("Solving RB problem via backslash")
  push!(RBVars.xₙ, RBVars.LHSₙ[1] \ RBVars.RHSₙ[1])

  return

end

function assemble_solve_reconstruct(
  FEMSpace::FOMST{1,D},
  RBInfo::ROMInfo{1},
  RBVars::ROM{1,T},
  μ::Vector{T}) where {D,T}

  assemble_RB_system(FEMSpace, RBInfo, RBVars, μ)
  RBVars.online_time += @elapsed begin
    solve_RB_system(RBVars)
  end
  reconstruct_FEM_solution(RBVars)

  return

end
