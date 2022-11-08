function check_saved_operators(
  RBInfo::ROMInfo{ID},
  Var::MVVariable{T}) where {ID,T}

  var = Var.var
  op = ""

  if var ∈ get_FEM_vectors(RBInfo) && RBInfo.online_RHS
    println("Vector $var will be built online: not importing its offline structures")
  else
    if !isfile(joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
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

function get_Φₛ(
  RBVars::ROM{ID,T},
  var::String) where {ID,T}

  if var ∈ ("B", "LB")
    Φₛ_left = RBVars.Φₛ[2]
  else
    Φₛ_left = RBVars.Φₛ[1]
  end
  Φₛ_right = RBVars.Φₛ[1]

  Φₛ_left, Φₛ_right

end

function get_Φₜ(
  RBVars::ROMMethodST{ID,T},
  var::String) where {ID,T}

  if var ∈ ("B", "LB")
    RBVars.Φₜ[2], RBVars.Φₜ[1]
  elseif var == "Bᵀ"
    RBVars.Φₜ[1], RBVars.Φₜ[2]
  else
    RBVars.Φₜ[1], RBVars.Φₜ[1]
  end

end

function get_Nₛ(
  RBVars::ROM{ID,T},
  var::String) where {ID,T}

  if var ∈ ("B", "LB")
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

function get_norms(solₕ)
  norms = ["H¹"]
  if length(solₕ) == 2
    push!(norms, "L²")
  end
  norms
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

function initialize_RB_system(RBVars::ROM{ID,T}) where {ID,T}
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
  RBVars.xₙ = Matrix{T}[]
  return
end

function initialize_online_time(RBVars::ROM{ID,T}) where {ID,T}
  RBVars.online_time = 0.0
  return
end

function VVariable(RBVars::ROM{ID,T}) where {ID,T}
  VVariable(RBVars.Vars)
end

function VVariable(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  args...) where {ID,T}

  VVariable(RBInfo, RBVars.Vars, args...)
end

function MVariable(RBVars::ROM{ID,T}) where {ID,T}

  MVariable(RBVars.Vars)
end

function MVariable(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  args...) where {ID,T}

  MVariable(RBInfo, RBVars.Vars, args...)
end

function assemble_termsₙ(
  Var::MVVariable{T},
  Param::ParamInfo) where T

  mult = Broadcasting(.*)
  sum(mult(Var.Matₙ, Param.θ))

end

function assemble_termsₙ(
  Vars::Vector{<:MVVariable{T}},
  Params::Vector{<:ParamInfo}) where T

  Broadcasting(assemble_termsₙ)(Vars, Params)

end

function assemble_termsₙ(
  Matₙ::Matrix{T},
  Φₜθ::Matrix{T}) where T

  kron(Matₙ, Φₜθ)

end

function assemble_function_termsₙ(
  Var::MVVariable{T},
  Param::ParamInfo) where T

  @assert Var.var == Param.var

  mult = Broadcasting(.*)
  termₙ(u) = sum(mult(Var.Matₙ, Param.fun(u)))

  termₙ

end

function assemble_function_termsₙ(
  Vars::Vector{<:MVVariable{T}},
  Params::Vector{<:ParamInfo}) where T

  Broadcasting(assemble_function_termsₙ)(Vars, Params)

end

function assemble_function_termsₙ(
  Matₙ::Matrix{T},
  Φₜθ::Function) where T

  termₙ(u) = kron(Matₙ, Φₜθ(u))
  termₙ

end

function Φₜ_by_Φₜ_by_θ(
  Φₜ_left::Matrix{T},
  Φₜ_right::Matrix{T},
  idx₁::UnitRange{Int},
  idx₂::UnitRange{Int},
  θ::Vector{Vector{T}}) where T

  nₜ_left, nₜ_right = size(Φₜ_left)[2], size(Φₜ_right)[2]

  ΦₜΦₜθ_fun(iₜ,jₜ,q) =
    sum(Φₜ_left[idx₁,iₜ] .* Φₜ_right[idx₂,jₜ] .* θ[q][idx₁])
  ΦₜΦₜθ_fun(jₜ,q) =
    Broadcasting(iₜ -> ΦₜΦₜθ_fun(iₜ,jₜ,q))(1:nₜ_left)
  ΦₜΦₜθ_fun(q) =
    Broadcasting(jₜ -> ΦₜΦₜθ_fun(jₜ,q))(1:nₜ_right)

  ϕₜϕₜθ_block = Broadcasting(ΦₜΦₜθ_fun)(eachindex(θ))
  Broadcasting(blocks_to_matrix)(ϕₜϕₜθ_block)::Vector{Matrix{T}}

end

function Φₜ_by_θ(
  Φₜ_left::Matrix{T},
  θ::Vector{Vector{T}}) where T

  nₜ_left = size(Φₜ_left)[2]

  Φₜθ_fun(iₜ,q) = sum(Φₜ_left[:,iₜ] .* θ[q])
  Φₜθ_fun(q) = reshape(Broadcasting(iₜ -> Φₜθ_fun(iₜ,q))(1:nₜ_left), :, 1)

  Broadcasting(Φₜθ_fun)(eachindex(θ))::Vector{Matrix{T}}

end

function Φₜ_by_Φₜ_by_θ(
  Φₜ_left::Matrix{T},
  Φₜ_right::Matrix{T},
  idx₁::UnitRange{Int},
  idx₂::UnitRange{Int},
  θ::Function) where T

  nₜ_left, nₜ_right = size(Φₜ_left)[2], size(Φₜ_right)[2]

  ΦₜΦₜθ_fun(u,iₜ,jₜ,q) =
    sum(Φₜ_left[idx₁,iₜ] .* Φₜ_right[idx₂,jₜ] .* θ(u)[q][idx₁])
  ΦₜΦₜθ_fun(u,jₜ,q) =
    Broadcasting(iₜ -> ΦₜΦₜθ_fun(u,iₜ,jₜ,q))(1:nₜ_left)
  ΦₜΦₜθ_fun(u,q) =
    Broadcasting(jₜ -> ΦₜΦₜθ_fun(u,jₜ,q))(1:nₜ_right)
  ΦₜΦₜθ_fun(u) =
    blocks_to_matrix(Broadcasting(q -> ΦₜΦₜθ_fun(u,q))(eachindex(θ)))

  ΦₜΦₜθ_fun::Function

end

function Φₜ_by_θ(
  Φₜ_left::Matrix{T},
  θ::Function) where T

  nₜ_left = size(Φₜ_left)[2]

  Φₜθ_fun(u,iₜ,q) = sum(Φₜ_left[:,iₜ] .* θ(u)[q])
  Φₜθ_fun(u,q) = reshape(Broadcasting(iₜ -> Φₜθ_fun(u,iₜ,q))(1:nₜ_left), :, 1)
  Φₜθ_fun(u) = Broadcasting(q -> Φₜθ_fun(u,q))(eachindex(θ))

  Φₜθ_fun::Function

end

function errors(
  solₕ::Matrix{T},
  sõl::Matrix{T},
  X::SparseMatrixCSC{Float,Int},
  norm::String) where T

  err_nb = compute_errors(solₕ, sõl, X)
  pointwise_err = abs.(solₕ - sõl)
  println("Online error, norm $norm: $err_nb")
  (err_nb, pointwise_err)::Tuple{Float64, Matrix{Float64}}
end

function errors(
  solsₕ::Vector{Matrix{T}},
  sõls::Vector{Matrix{T}},
  Xs::Vector{SparseMatrixCSC{Float,Int}},
  norms::Vector{String}) where T

  errs = Broadcasting(errors)(solsₕ, sõls, Xs, norms)
  errs::Vector{Tuple{Float64, Matrix{Float64}}}
end

function errors(
  solsₕ::Vector{Vector{Matrix{T}}},
  sõls::Vector{Vector{Matrix{T}}},
  Xs::Vector{SparseMatrixCSC{Float,Int}},
  norms::Vector{String}) where T

  errs = Broadcasting((solₕ, sõl) -> errors(solₕ, sõl, Xs, norms))(solsₕ, sõls)
  errs::Vector{Vector{Tuple{Float64, Matrix{Float64}}}}
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

  @assert size(xₕ)[2] == size(x̃)[2] "Something is wrong"

  Nₜ = size(xₕ)[2]

  function normᵢ(i::Int)
    LinearAlgebra.norm(xₕ[:, i] - x̃[:, i], X), LinearAlgebra.norm(xₕ[:, i], X)
  end

  norms = Broadcasting(normᵢ)(1:Nₜ)::Vector{Tuple{T,T}}
  norm_err, norm_sol = first.(norms), last.(norms)

  norm(norm_err) / norm(norm_sol)

end

function Base.isempty(Var::MVVariable{T}) where T
  isempty(Var.var)
end

function Base.isempty(Param::ParamInfo)
  isempty(Param.var)
end

function get_nonempty_Vars(Vars::Vector{<:MVVariable{T}}) where T
  Vars[.!Broadcasting(isempty)(Vars)]
end
