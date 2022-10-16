function check_saved_operators(
  RBInfo::ROMInfo{ID},
  Var::MVVariable{T}) where {ID,T}

  var = Var.var
  op = ""

  if var ∈ get_FEM_vectors(RBInfo) && RBInfo.online_RHS
    println("Vector $var will be built online: not importing its offline structures")
  else
    if isfile(joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
    else
      println("Failed to import offline structures for $var: must build them")
      op = var
    end
  end

  op::String

end

function check_saved_operators(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> check_saved_operators(RBInfo, Var))(Vars)

end

function get_Φₛ(RBVars::RB, var::String)
  if var ∈ ("B", "Lc")
    Φₛ_left = RBVars.Φₛ[2]
  else
    Φₛ_left = RBVars.Φₛ[1]
  end
  Φₛ_right = RBVars.Φₛ[1]
  Φₛ_left, Φₛ_right
end

function get_Nₛ(RBVars::RB, var::String)
  if var ∈ ("B", "Lc")
    RBVars.Nₛ[2]
  else
    RBVars.Nₛ[1]
  end
end

function get_blocks_position(::ROMInfo{ID}) where ID
  if ID == 1
    ([1], [1])
  else
    ([1, 2, 3], [1, 2])
  end
end

function times_dictionary(
  RBInfo::ROMInfo,
  offline_time::Float,
  online_time::Float)

  if RBInfo.get_offline_structures
    offline_time = NaN
  end

  Dict("off_time"=>offline_time, "on_time"=>online_time)

end

function initialize_RB_system(RBVars::RB{T}) where T
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
  RBVars.xₙ = Matrix{T}[]
  return
end

function initialize_online_time(RBVars::RB)
  RBVars.online_time = 0.0
  return
end

function VVariable(RBVars::RB{T}) where T
  VVariable(RBVars.Vars)
end

function VVariable(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  args...) where {ID,T}

  VVariable(RBInfo, RBVars.Vars, args...)
end

function MVariable(RBVars::RB{T}) where T

  MVariable(RBVars.Vars)
end

function MVariable(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  args...) where {ID,T}

  MVariable(RBInfo, RBVars.Vars, args...)
end

function assemble_termsₙ(
  Var::MVVariable{T},
  Param::ParamInfo) where T

  @assert Var.var == Param.var

  mult = Broadcasting(.*)
  Var.var ∈ ("L", "Lc") ? -sum(mult(Var.Matₙ, Param.θ)) : sum(mult(Var.Matₙ, Param.θ))

end

function assemble_termsₙ(
  Vars::Vector{<:MVVariable{T}},
  Params::Vector{<:ParamInfo}) where T

  Broadcasting(assemble_termsₙ)(Vars, Params)

end

function compute_errors(
  xₕ::Vector{T},
  x̃::Vector{T},
  X::SparseMatrixCSC{T,Int}) where T

  LinearAlgebra.norm(xₕ - x̃, X) / LinearAlgebra.norm(xₕ, X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::SparseMatrixCSC{T,Int}) where T

  @assert size(xₕ)[2] == size(x̃)[2] == 1 "Something is wrong"
  compute_errors(xₕ[:, 1], x̃[:, 1], X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::SparseMatrixCSC{T,Int},
  Nₜ::Int) where T

  function normᵢ(i::Int)
    LinearAlgebra.norm(xₕ[:, i] - x̃[:, i], X), LinearAlgebra.norm(xₕ[:, i], X)
  end

  norms = Broadcasting(normᵢ)(1:Nₜ)::Vector{Tuple{Int,Int}}
  norm_err, norm_sol = first!(norms), last!(norms)

  norm_err ./ norm_sol, norm(norm_err) / norm(norm_sol)

end
