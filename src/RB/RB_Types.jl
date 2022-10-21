struct ROMPath
  ROM_structures_path::String
  results_path::String
end

function ROMPath(FEMPaths)

  ROM_path = joinpath(FEMPaths.current_test, "ROM")
  create_dir(ROM_path)
  ROM_structures_path = joinpath(ROM_path, "ROM_structures")
  create_dir(ROM_structures_path)
  results_path = joinpath(ROM_path, "results")
  create_dir(results_path)

  ROMPath(ROM_structures_path, results_path)

end

abstract type ROMInfo{ID} end

struct ROMInfoS{ID} <: ROMInfo{ID}
  FEMInfo::FOMInfoS{ID}
  Paths::ROMPath
  nₛ::Int
  ϵₛ::Float
  use_norm_X::Bool
  online_RHS::Bool
  nₛ_MDEIM::Int
  post_process::Bool
  get_offline_structures::Bool
  save_offline::Bool
  save_online::Bool
end

mutable struct ROMInfoST{ID} <: ROMInfo{ID}
  FEMInfo::FOMInfoST{ID}
  Paths::ROMPath
  nₛ::Int
  ϵₛ::Float
  ϵₜ::Float
  use_norm_X::Bool
  t_red_method::String
  online_RHS::Bool
  nₛ_MDEIM::Int
  nₛ_MDEIM_time::Int
  post_process::Bool
  get_offline_structures::Bool
  save_offline::Bool
  save_online::Bool
  st_MDEIM::Bool
  functional_MDEIM::Bool
  adaptivity::Bool
end

function Base.getproperty(RBInfo::ROMInfoS, sym::Symbol)
  if sym in (:affine_structures, :unknowns, :structures)
    getfield(RBInfo.FEMInfo, sym)::Vector{String}
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)::String
  else
    getfield(RBInfo, sym)
  end
end

function Base.getproperty(RBInfo::ROMInfoST, sym::Symbol)
  if sym in (:affine_structures, :unknowns, :structures)
    getfield(RBInfo.FEMInfo, sym)::Vector{String}
  elseif sym in (:θ, :t₀, :tₗ, :δt)
      getfield(RBInfo.FEMInfo, sym)::Float
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)::String
  else
    getfield(RBInfo, sym)
  end
end

abstract type MVMDEIM{T} end
abstract type MVVariable{T} end

mutable struct VMDEIM{T} <: MVMDEIM{T}
  Mat::Matrix{T}
  Matᵢ::Matrix{T}
  idx::Vector{Int}
  time_idx::Vector{Int}
  el::Vector{Int}
end

mutable struct MMDEIM{T} <: MVMDEIM{T}
  Mat::Matrix{T}
  Matᵢ::Matrix{T}
  idx::Vector{Int}
  time_idx::Vector{Int}
  row_idx::Vector{Int}
  el::Vector{Int}
end

mutable struct VVariable{T} <: MVVariable{T}
  var::String
  Matₙ::Vector{Matrix{T}}
  MDEIM::VMDEIM{T}
end


mutable struct MVariable{T} <: MVVariable{T}
  var::String
  Matₙ::Vector{Matrix{T}}
  MDEIM::MMDEIM{T}
end

function VVariable(var::String, ::Type{T}) where T

  Matₙ = Matrix{T}[]

  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  MDEIM_Vec = VMDEIM{T}(Mat, Matᵢ, idx, time_idx, el)

  VVariable{T}(var, Matₙ, MDEIM_Vec)

end

function VVariable(RBInfo::ROMInfo{ID}, var::String, ::Type{T}) where {ID,T}

  if isempty(var)
    VVariable(var, T)::VVariable{T}
  else
    if isvector(RBInfo, var)
      VVariable(var, T)::VVariable{T}
    else
      error("Unrecognized variable")
    end
  end

end

function MVariable(var::String, ::Type{T}) where T

  Matₙ = Matrix{T}[]

  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  row_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  MDEIM_Mat = MMDEIM{T}(Mat, Matᵢ, idx, time_idx, row_idx, el)

  MVariable{T}(var, Matₙ, MDEIM_Mat)

end

function MVariable(RBInfo::ROMInfo{ID}, var::String, ::Type{T}) where {ID,T}

  if isempty(var)
    MVariable(var, T)::MVariable{T}
  else
    if ismatrix(RBInfo, var)
      MVariable(var, T)::MVariable{T}
    else
      error("Unrecognized variable")
    end
  end

end

function VVariable(Vars::Vector{<:MVVariable{T}}) where T
  Vars_to_get = findall(x->typeof(x) == VVariable{T}, Vars)
  Broadcasting(idx->getindex(Vars, idx))(Vars_to_get)::Vector{VVariable{T}}
end

function VVariable(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}},
  var::String) where {ID,T}

  if isempty(var)
    VV = MVariable(RBInfo, "", T)
  else
    @assert isvector(RBInfo, var)
    Var_to_get = findall(x->x.var == var, Vars)[1]
    VV = Vars[Var_to_get]
  end

  VV::VVariable{T}

end

function VVariable(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}},
  vars::Vector{String}) where {ID,T}

  if isempty(vars)
    VV = [VVariable(RBInfo, "", T)]
  else
    VV = Broadcasting(var->VVariable(RBInfo, Vars, var))(vars)
  end

  VV::Vector{VVariable{T}}

end

function VVariable(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  operators = intersect(get_FEM_vectors(RBInfo), set_operators(RBInfo))
  VVariable(RBInfo, Vars, operators)::Vector{VVariable{T}}

end

function MVariable(Vars::Vector{<:MVVariable{T}}) where T
  Vars_to_get = findall(x->typeof(x) == MVariable{T}, Vars)
  Broadcasting(idx->getindex(Vars, idx))(Vars_to_get)::Vector{MVariable{T}}
end

function MVariable(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}},
  var::String) where {ID,T}

  if isempty(var)
    MV = MVariable(RBInfo, "", T)
  else
    @assert ismatrix(RBInfo, var)
    Var_to_get = findall(x->x.var == var, Vars)[1]
    MV = Vars[Var_to_get]
  end

  MV::MVariable{T}

end

function MVariable(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}},
  vars::Vector{String}) where {ID,T}

  if isempty(vars)
    MV = [MVariable(RBInfo, "", T)]
  else
    MV = Broadcasting(var->MVariable(RBInfo, Vars, var))(vars)
  end

  MV::Vector{MVariable{T}}

end

function MVariable(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  operators = get_FEM_matrices(RBInfo)
  MVariable(RBInfo, Vars, operators)::Vector{MVariable{T}}

end

abstract type ROM{ID,T} end
abstract type ROMS{ID,T} <: ROM{ID,T} end
abstract type ROMST{ID,T} <: ROM{ID,T} end

mutable struct VarsS{T} <: ROMS{Any,T}
  S::Vector{Matrix{T}}
  Φₛ::Vector{Matrix{T}}
  x̃::Vector{Vector{Matrix{T}}}
  X₀::Vector{SparseMatrixCSC{Float, Int}}
  Nₛ::Vector{Int}
  nₛ::Vector{Int}
  offline_time::Float
  online_time::Float
end

function VarsS(::Type{T}) where T

  S = Matrix{T}[]
  Φₛ = Matrix{T}[]
  x̃ = Vector{Matrix{T}}[]
  X₀ = Matrix{SparseMatrixCSC{Float, Int}}[]
  Nₛ = Int[]
  nₛ = Int[]
  offline_time = 0.0
  online_time = 0.0

  VarsS{T}(S, Φₛ, x̃, X₀, Nₛ, nₛ, offline_time, online_time)

end

mutable struct ROMMethodS{ID,T} <: ROMS{ID,T}
  SV::VarsS{T}
  Vars::Vector{MVVariable{T}}
  xₙ::Vector{Matrix{T}}
  LHSₙ::Vector{Matrix{T}}
  RHSₙ::Vector{Matrix{T}}
end

function ROMMethodS(RBInfo::ROMInfoS{ID}, ::Type{T}) where {ID,T}
  SV = VarsS(T)

  FEM_matrices = get_FEM_matrices(RBInfo)
  MVars = Broadcasting(var->MVariable(RBInfo, var, T))(FEM_matrices)
  FEM_vectors = get_FEM_vectors(RBInfo)
  VVars = Broadcasting(var->VVariable(RBInfo, var, T))(FEM_vectors)
  Vars = vcat(MVars, VVars)::Vector{MVVariable{T}}

  xₙ = Matrix{T}[]
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]

  ROMMethodS{ID,T}(SV, Vars, xₙ, LHSₙ, RHSₙ)
end

function Base.getproperty(RBVars::ROMMethodS{ID,T}, sym::Symbol) where {ID,T}
  if sym ∈ (:S, :Φₛ)
    getfield(RBVars.SV, sym)::Vector{Matrix{T}}
  elseif sym == :x̃
    getfield(RBVars.SV, sym)::Vector{Vector{Matrix{T}}}
  elseif sym == :X₀
    getfield(RBVars.SV, sym)::Vector{SparseMatrixCSC{Float, Int}}
  elseif sym ∈ (:Nₛ, :nₛ)
    getfield(RBVars.SV, sym)::Vector{Int}
  elseif sym ∈ (:offline_time, :online_time)
    getfield(RBVars.SV, sym)::Float
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::ROMMethodS{ID,T}, sym::Symbol, x) where {ID,T}
  if sym ∈ (:S, :Φₛ)
    setfield!(RBVars.SV, sym, x)::Vector{Matrix{T}}
  elseif sym == :x̃
    setfield!(RBVars.SV, sym, x)::Vector{Vector{Matrix{T}}}
  elseif sym == :X₀
    setfield!(RBVars.SV, sym, x)::Vector{SparseMatrixCSC{Float, Int}}
  elseif sym ∈ (:Nₛ, :nₛ)
    setfield!(RBVars.SV, sym, x)::Vector{Int}
  elseif sym ∈ (:offline_time, :online_time)
    setfield!(RBVars.SV, sym, x)::Float
  else
    setfield!(RBVars, sym, x)
  end
end

mutable struct VarsT{T} <: ROMST{Any,T}
  Φₜ::Vector{Matrix{T}}
  Nₜ::Int
  nₜ::Vector{Int}
  N::Vector{Int}
  n::Vector{Int}
end

function VarsT(::Type{T}) where T

  Φₜ = Matrix{T}[]
  Nₜ = 0
  nₜ = Int[]
  N = Int[]
  n = Int[]

  VarsT{T}(Φₜ, Nₜ, nₜ, N, n)

end

mutable struct ROMMethodST{ID,T} <: ROMST{ID,T}
  SV::VarsS{T}
  TV::VarsT{T}
  Vars::Vector{MVVariable{T}}
  xₙ::Vector{Matrix{T}}
  LHSₙ::Vector{Matrix{T}}
  RHSₙ::Vector{Matrix{T}}
end

function ROMMethodST(RBInfo::ROMInfoST{ID}, ::Type{T}) where {ID,T}
  SV = VarsS(T)
  TV = VarsT(T)

  FEM_matrices = get_FEM_matrices(RBInfo)
  MVars = Broadcasting(var->MVariable(RBInfo, var, T))(FEM_matrices)
  FEM_vectors = get_FEM_vectors(RBInfo)
  VVars = Broadcasting(var->VVariable(RBInfo, var, T))(FEM_vectors)
  Vars = vcat(MVars, VVars)::Vector{MVVariable{T}}

  xₙ = Matrix{T}[]
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]

  ROMMethodST{ID,T}(SV, TV, Vars, xₙ, LHSₙ, RHSₙ)
end

function Base.getproperty(RBVars::ROMMethodST{ID,T}, sym::Symbol) where {ID,T}
  if sym ∈ (:S, :Φₛ)
    getfield(RBVars.SV, sym)::Vector{Matrix{T}}
  elseif sym == :x̃
    getfield(RBVars.SV, sym)::Vector{Vector{Matrix{T}}}
  elseif sym == :X₀
    getfield(RBVars.SV, sym)::Vector{SparseMatrixCSC{Float, Int}}
  elseif sym ∈ (:Nₛ, :nₛ)
    getfield(RBVars.SV, sym)::Vector{Int}
  elseif sym ∈ (:offline_time, :online_time)
    getfield(RBVars.SV, sym)::Float
  elseif sym == :Φₜ
    getfield(RBVars.TV, sym)::Vector{Matrix{T}}
  elseif sym == :Nₜ
    getfield(RBVars.TV, sym)::Int
  elseif sym ∈ (:nₜ, :N, :n)
    getfield(RBVars.TV, sym)::Vector{Int}
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::ROMMethodST{ID,T}, sym::Symbol, x) where {ID,T}
  if sym ∈ (:S, :Φₛ)
    setfield!(RBVars.SV, sym, x)::Vector{Matrix{T}}
  elseif sym == :x̃
    setfield!(RBVars.SV, sym, x)::Vector{Vector{Matrix{T}}}
  elseif sym == :X₀
    setfield!(RBVars.SV, sym, x)::Vector{SparseMatrixCSC{Float, Int}}
  elseif sym ∈ (:Nₛ, :nₛ)
    setfield!(RBVars.SV, sym, x)::Vector{Int}
  elseif sym ∈ (:offline_time, :online_time)
    setfield!(RBVars.SV, sym, x)::Float
  elseif sym == :Φₜ
    setfield!(RBVars.TV, sym, x)::Vector{Matrix{T}}
  elseif sym == :Nₜ
    setfield!(RBVars.TV, sym, x)::Int
  elseif sym ∈ (:nₜ, :N, :n)
    setfield!(RBVars.TV, sym, x)::Vector{Int}
  else
    setfield!(RBVars, sym, x)
  end
end
