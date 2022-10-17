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
  use_norm_X::Bool
  online_RHS::Bool
  nₛ_MDEIM::Int
  nₛ_MDEIM_time::Int
  post_process::Bool
  get_offline_structures::Bool
  save_offline::Bool
  save_online::Bool
  time_reduction_technique::String
  t₀::Float
  tₗ::Float
  δt::Float
  θ::Float
  ϵₜ::Float
  st_MDEIM::Bool
  functional_MDEIM::Bool
  adaptivity::Bool
end

function Base.getproperty(RBInfo::ROMInfo, sym::Symbol)
  if sym in (:affine_structures, :unknowns, :structures)
    getfield(RBInfo.FEMInfo, sym)::Vector{String}
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
  x̃::Vector{Matrix{T}}
  X₀::Vector{SparseMatrixCSC{Float, Int}}
  Nₛ::Vector{Int}
  nₛ::Vector{Int}
  offline_time::Float
  online_time::Float
end

function VarsS(::Type{T}) where T

  S = Matrix{T}[]
  Φₛ = Matrix{T}[]
  x̃ = Matrix{T}[]
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
  if sym ∈ (:S, :Φₛ, :x̃)
    getfield(RBVars.SV, sym)::Vector{Matrix{T}}
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
  if sym ∈ (:S, :Φₛ, :x̃)
    setfield!(RBVars.SV, sym, x)::Vector{Matrix{T}}
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

#= function setup_RBVars(RBInfo::ROMInfoS{ID}, ::Type{T}) where {ID,T}
  if ID == 1
    PoissonS(RBInfo, T)
  elseif ID == 2
    StokesS(RBInfo, T)
  elseif ID == 3
    NavierStokesS(RBInfo, T)
  else
    error("Not implemented")
  end
end

function setup_RBVars(RBInfo::ROMInfoST{ID}, ::Type{T}) where {ID,T}
  if ID == 1
    PoissonST(RBInfo, T)
  elseif ID == 2
    StokesST(RBInfo, T)
  elseif ID == 3
    NavierStokesST(RBInfo, T)
  else
    error("Not implemented")
  end
end =#

#= mutable struct PoissonST{T} <: ROMST{ID,T}
  Steady::PoissonS{T}; Φₜᵘ::Vector{Matrix{T}}; Mₙ::Vector{Matrix{T}}; MDEIM_M::MMDEIM{T};
  Nₜ::Int; N::Vector{Int}; nₜ::Vector{Int}; n::Vector{Int}
end

mutable struct StokesS{T} <: ROMS{ID,T}
  Poisson::PoissonS{T}; Bₙ::Vector{Matrix{T}}; Lcₙ::Vector{Matrix{T}};
  MDEIM_B::MMDEIM{T}; MDEIM_Lc::VMDEIM{T};
end

mutable struct StokesST{T} <: ROMST{ID,T}
  Poisson::PoissonST{T}; Steady::StokesS{T}
end

mutable struct NavierStokesS{T} <: ROMS{ID,T}
  Stokes::StokesS{T}; Cₙ::Vector{Matrix{T}}; Dₙ::Vector{Matrix{T}};
  MDEIM_C::MMDEIM{T}; MDEIM_D::MMDEIM{T}
end

mutable struct NavierStokesST{T} <: ROMST{ID,T}
  Stokes::StokesST{T}; Steady::NavierStokesS{T};
end

function setup_RBVars(NT::NTuple{2,Int}, ::Type{T}) where T

  PoissonST{T}(
    setup_RBVars(NTuple(1, Int), T), init_RBVars(NT, T)...)

end

function setup_RBVars(NT::NTuple{3,Int}, ::Type{T}) where T

  StokesS{T}(
    setup_RBVars(NTuple(1, Int), T), init_RBVars(NT, T)...)

end

function setup_RBVars(NT::NTuple{4,Int}, ::Type{T}) where T

  StokesST{T}(
    setup_RBVars(NTuple(2, Int), T), setup_RBVars(NTuple(3, Int), T))

end

function setup_RBVars(NT::NTuple{5,Int}, ::Type{T}) where T

  NavierStokesS{T}(
    setup_RBVars(NTuple(3, Int), T), init_RBVars(NT, T)...)

end

function setup_RBVars(::NTuple{6,Int}, ::Type{T}) where T

  NavierStokesST{T}(
    setup_RBVars(NTuple(4, Int), T), setup_RBVars(NTuple(5, Int), T))

end

function Base.getproperty(RBVars::PoissonST, sym::Symbol)
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Steady, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::PoissonST, sym::Symbol, x::T) where T
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesS, sym::Symbol)
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesS, sym::Symbol, x::T) where T
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Poisson, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesST, sym::Symbol)
  if sym in (:Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc)
    getfield(RBVars.Steady, sym)
  elseif sym in (:Φₜ, :Mₙ, :MDEIM_M, :nₜ, :Nₜ, :N, :n)
    getfield(RBVars.Poisson, sym)
  elseif sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Poisson.Steady, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesST, sym::Symbol, x::T) where T
  if sym in (:Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Φₜ, :Mₙ, :MDEIM_M, :nₜ, :Nₜ, :N, :n)
    setfield!(RBVars.Poisson, sym, x)::T
  elseif sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Poisson.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesS, sym::Symbol)
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Stokes.Poisson, sym)
  elseif sym in (:Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc)
    getfield(RBVars.Stokes, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::NavierStokesS, sym::Symbol, x::T) where T
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Stokes.Poisson, sym, x)::T
  elseif sym in (:Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesST, sym::Symbol)
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛ, :p̃, :pₙ, :p̂, :Xu, :Xp₀, :Nₛᵖ, :nₛᵖ)
    getfield(RBVars.Steady, sym)
  elseif sym in (:Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_B, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Φₜᵖ, :Nᵖ, :nₜᵖ, :nᵖ)
    getfield(RBVars.Stokes, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::NavierStokesST, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xu₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛ, :p̃, :pₙ, :p̂, :Xu, :Xp₀, :Nₛᵖ, :nₛᵖ)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_B, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Φₜᵖ, :Nᵖ, :nₜᵖ, :nᵖ)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end =#


#= function init_RBVars(::NTuple{2,Int}, ::Type{T}) where T

  Φₜ = Matrix{T}[]
  Mₙ = Matrix{T}[]
  MDEIM_M = MMDEIM(init_MMDEIM(T)...)
  Nₜ = 0
  N = 0
  nₜ = 0
  n = 0

  Φₜ, Mₙ, MDEIM_M, Nₜ, N, nₜ, n

end

function init_RBVars(::NTuple{3,Int}, ::Type{T}) where T

  Bₙ = Matrix{T}[]
  Lcₙ = Matrix{T}[]
  MDEIM_B = MMDEIM(init_MMDEIM(T)...)
  MDEIM_Lc = VMDEIM(init_VMDEIM(T)...)

  Bₙ, Lcₙ, MDEIM_B, MDEIM_Lc

end

function init_RBVars(::NTuple{4,Int}, ::Type{T}) where T
  nothing
end

function init_RBVars(::NTuple{5,Int}, ::Type{T}) where T

  Cₙ = Matrix{T}[]
  Dₙ = Matrix{T}[]
  MDEIM_C = MMDEIM(init_MMDEIM(T)...)
  MDEIM_D = MMDEIM(init_MMDEIM(T)...)

  Cₙ, Dₙ, MDEIM_C, MDEIM_D

end
 =#
