function assemble_LHSₙ(
  RBInfo::Info,
  RBVars::PoissonS{T},
  Params::Vector{ParamInfoS}) where T

  Matsₙ = assemble_matricesₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, Matsₙ[1])

end

function assemble_RHSₙ(
  RBInfo::Info,
  RBVars::PoissonS{T},
  Params::Vector{ParamInfoS}) where T

  Vecsₙ = assemble_vectorsₙ(RBInfo, RBVars, Params)::Vector{Matrix{T}}
  push!(RBVars.RHSₙ, sum(Vecsₙ))

end

function assemble_RHSₙ(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::PoissonS{T},
  μ::Vector{T}) where T

  ParamF = ParamInfo(RBInfo, μ, "F")
  ParamH = ParamInfo(RBInfo, μ, "H")
  ParamL = ParamInfo(RBInfo, μ, "L")

  F = assemble_FEM_structure(FEMSpace, RBInfo, ParamF)
  H = assemble_FEM_structure(FEMSpace, RBInfo, ParamH)
  L = assemble_FEM_structure(FEMSpace, RBInfo, ParamL)

  push!(RBVars.RHSₙ, reshape(RBVars.Φₛ[1]' * (F + H - L), :, 1)::Matrix{T})

end

function solve_RB_system(RBVars::PoissonS{T}) where T

  println("Solving RB problem via backslash")

  RBVars.online_time += @elapsed begin
    push!(RBVars.xₙ, RBVars.LHSₙ[1] \ RBVars.RHSₙ[1])
  end

end
