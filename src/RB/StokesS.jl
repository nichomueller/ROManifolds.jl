function assemble_LHSₙ(
  RBInfo::ROMInfoS{2},
  RBVars::ROMMethodS{2,T},
  Params::Vector{ParamInfoS}) where T

  Matsₙ = assemble_matricesₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  LHSₙ = vcat(hcat(Matsₙ[1], Matsₙ[2]), hcat(Matsₙ[3],
    zeros(T, RBVars.nₛ[2], RBVars.nₛ[2])))
  push!(RBVars.LHSₙ, LHSₙ)

  return

end

function assemble_RHSₙ(
  RBInfo::ROMInfoS{2},
  RBVars::ROMMethodS{2,T},
  Params::Vector{ParamInfoS}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  RHSₙ = vcat(sum(Vecsₙ[1:3]), Vecsₙ[end])
  push!(RBVars.RHSₙ, RHSₙ)

  return

end

function assemble_RHSₙ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{2},
  RBVars::ROMMethodS{2,T},
  μ::Vector{T}) where {D,T}

  RHS = assemble_RHS(FEMSpace, RBInfo, μ)
  RHS = vcat(sum(RHS[1:3]), RHS[end])
  push!(RBVars.RHSₙ, reshape(RBVars.Φₛ[1]' * RHS, :, 1)::Matrix{T})

  return

end

function solve_RB_system(RBVars::ROMMethodS{2,T}) where T

  println("Solving RB problem via backslash")
  push!(RBVars.xₙ, RBVars.LHSₙ[1] \ RBVars.RHSₙ[1])

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
