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
end

function initialize_online_time(RBVars::RB)
  RBVars.online_time = 0.0
end

function assemble_termsₙ(
  Vars::Vector{<:MVVariable},
  Params::Vector{<:ParamInfo},
  operators::Vector{String})

  mult = Broadcasting(.*)

  function assemble_termₙ(var::String)
    Var = MVVariable(Vars, var)
    Param = ParamInfo(Params, var)
    if var ∈ ("L", "Lc")
      -sum(mult(Var.Matₙ, Param.θ))
    else
      sum(mult(Var.Matₙ, Param.θ))
    end
  end

  Broadcasting(assemble_termₙ)(operators)

end

function compute_errors(
  xₕ::Vector{T},
  x̃::Vector{T},
  X::Matrix{T}) where T

  norm(xₕ - x̃, X) / norm(xₕ, X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::Matrix{T}) where T

  @assert size(xₕ)[2] == size(x̃)[2] == 1 "Something is wrong"
  compute_errors(xₕ[:, 1], x̃[:, 1], X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::Matrix{T},
  Nₜ::Int) where T

  norm_err = zeros(T, Nₜ)
  norm_sol = zeros(T, Nₜ)

  @simd for i = 1:Nₜ
    norm_err[i] = norm(xₕ[:, i] - x̃[:, i], X)
    norm_sol[i] = norm(xₕ[:, i], X)
  end

  norm_err ./ norm_sol, norm(norm_err) / norm(norm_sol)

end
