################################## ONLINE ######################################

function assemble_LHSₙ(
  RBInfo::ROMInfoS{2},
  RBVars::ROMMethodS{2,T},
  Params::Vector{ParamInfoS}) where T

  Matsₙ = assemble_matricesₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, Matsₙ[1])
  push!(RBVars.LHSₙ, Matsₙ[2])

  return

end

function assemble_RHSₙ(
  RBInfo::ROMInfoS{2},
  RBVars::ROMMethodS{2,T},
  Params::Vector{ParamInfoS}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.RHSₙ, sum(Vecsₙ[1:end-1]))
  push!(RBVars.RHSₙ, Vecsₙ[end])

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{2},
  RBVars::ROMMethodS{2,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  push!(RBVars.RHSₙ, RBVars.Φₛ[1]' * sum(RHS[1:end-1]))
  push!(RBVars.RHSₙ, RBVars.Φₛ[2]' * RHS[end])

  return

end

function assemble_RB_system(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{2},
  RBVars::ROM{2,T},
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

function solve_RB_system(RBVars::ROMMethodS{2,T}) where T

  println("Solving RB problem via backslash")
  LHSₙ = vcat(hcat(RBVars.LHSₙ[1], Matrix{T}(-RBVars.LHSₙ[2]')),
    hcat(RBVars.LHSₙ[2], zeros(T, RBVars.nₛ[2], RBVars.nₛ[2])))
  RHSₙ = vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2])

  xₙ = LHSₙ \ RHSₙ
  push!(RBVars.xₙ, xₙ[1:RBVars.nₛ[1],:])
  push!(RBVars.xₙ, xₙ[RBVars.nₛ[1]+1:end,:])

  return

end

function assemble_solve_reconstruct(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{2},
  RBVars::ROM{2,T},
  μ::Vector{T}) where {D,T}

  assemble_RB_system(FEMSpace, RBInfo, RBVars, μ)
  RBVars.online_time += @elapsed begin
    solve_RB_system(RBVars)
  end
  reconstruct_FEM_solution(RBVars)

  return

end
