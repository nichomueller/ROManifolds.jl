function assemble_LHSₙ(
  RBInfo::ROMInfoS{1},
  RBVars::ROMMethodS{1,T},
  Params::Vector{ParamInfoS}) where T

  Matsₙ = assemble_matricesₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, Matsₙ[1])

  return

end

function assemble_RHSₙ(
  RBInfo::ROMInfoS{1},
  RBVars::ROMMethodS{1,T},
  Params::Vector{ParamInfoS}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.RHSₙ, sum(Vecsₙ))

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{1},
  RBVars::ROMMethodS{1,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  push!(RBVars.RHSₙ, reshape(RBVars.Φₛ[1]' * sum(RHS), :, 1)::Matrix{T})

  return

end

function solve_RB_system(RBVars::ROMMethodS{1,T}) where T

  println("Solving RB problem via backslash")
  push!(RBVars.xₙ, RBVars.LHSₙ[1] \ RBVars.RHSₙ[1])

  return

end

function assemble_solve_reconstruct(
  FEMSpace::FOM{D},
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
