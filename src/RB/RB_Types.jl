abstract type RB{T} end
abstract type RBS{T} <: RB end
abstract type RBST{T} <: RB end

abstract type MVMDEIM{T} end
abstract type MVVariable{T} end

abstract type ROMInfo end

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
  MDEIM_Vec = VMDEIM(Mat, Matᵢ, idx, time_idx, el)

  VVariable(var, Matₙ, MDEIM_Vec)

end

function MVariable(var::String, ::Type{T}) where T

  Matₙ = Matrix{T}[]

  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  row_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  MDEIM_Mat = MMDEIM(Mat, Matᵢ, idx, time_idx, row_idx, el)

  MVariable(var, Matₙ, MDEIM_Mat)

end

function MVVariable(RBInfo::ROMInfo, var::String, ::Type{T}) where T

  if var ∈ get_FEM_vectors(RBInfo)
    return VVariable(var, T)
  elseif var ∈ get_FEM_matrices(RBInfo)
    return MVariable(var, T)
  else
    error("Unrecognized variable")
  end

end

function MVVariable(Vars::Vector{MVVariable}, var::String)

  for Var in Vars
    if Var.var == var
      return Var
    end
  end

  error("Unrecognized variable")

end

mutable struct SteadyVariables{T} <: RBS{T}
  S::Vector{Matrix{T}}
  Φₛ::Vector{Matrix{T}}
  x̃::Vector{Matrix{T}}
  X₀::Vector{SparseMatrixCSC{Float, Int}}
  Nₛ::Vector{Int}
  nₛ::Vector{Int}
  offline_time::Float
  online_time::Float
end

function SteadyVariables(::Type{T}) where T

  S = Matrix{T}[]
  Φₛ = Matrix{T}[]
  x̃ = Matrix{T}[]
  X₀ = Matrix{SparseMatrixCSC{Float, Int}}[]
  Nₛ = Int[]
  nₛ = Int[]
  offline_time = 0.0
  online_time = 0.0

  SteadyVariables{T}(S, Φₛ, x̃, X₀, Nₛ, nₛ, offline_time, online_time)

end

mutable struct PoissonS{T} <: RBS{T}
  SV::SteadyVariables{T}
  Vars::Vector{MVVariable{T}}
  xₙ::Vector{Matrix{T}}
  LHSₙ::Vector{Matrix{T}}
  RHSₙ::Vector{Matrix{T}}
end

function PoissonS(RBInfo::ROMInfoS, ::Type{T}) where T
  #= VarA = MVariable("A", T)
  VarF = VVariable("F", T)
  VarH = VVariable("H", T)
  VarL = VVariable("L", T)
  Vars = [VarA, VarF, VarH, VarL] =#
  SV = SteadyVariables(T)
  MVVars(var) =  MVVariable(RBInfo, var, T)
  Vars = Broadcasting(MVVars)(RBInfo.structures)
  xₙ = Matrix{T}[]
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]

  PoissonS{T}(SV, Vars, xₙ, LHSₙ, RHSₙ)

end

function setup_RBVars(::NTuple{1,Int}, RBInfo::ROMInfo, ::Type{T}) where T
  PoissonS(RBInfo, T)
end

function Base.getproperty(RBVars::PoissonS, sym::Symbol)
  if sym in (:S, :Φₛ, :x̃, :X₀, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.SV, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::PoissonS, sym::Symbol, x::T) where T
  if sym in (:S, :Φₛ, :x̃, :X₀, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.SV, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

#= mutable struct PoissonST{T} <: RBST{T}
  Steady::PoissonS{T}; Φₜᵘ::Vector{Matrix{T}}; Mₙ::Vector{Matrix{T}}; MDEIM_M::MMDEIM{T};
  Nₜ::Int; N::Vector{Int}; nₜ::Vector{Int}; n::Vector{Int}
end

mutable struct StokesS{T} <: RBS{T}
  Poisson::PoissonS{T}; Bₙ::Vector{Matrix{T}}; Lcₙ::Vector{Matrix{T}};
  MDEIM_B::MMDEIM{T}; MDEIM_Lc::VMDEIM{T};
end

mutable struct StokesST{T} <: RBST{T}
  Poisson::PoissonST{T}; Steady::StokesS{T}
end

mutable struct NavierStokesS{T} <: RBS{T}
  Stokes::StokesS{T}; Cₙ::Vector{Matrix{T}}; Dₙ::Vector{Matrix{T}};
  MDEIM_C::MMDEIM{T}; MDEIM_D::MMDEIM{T}
end

mutable struct NavierStokesST{T} <: RBST{T}
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

struct ROMPath
  ROM_structures_path::String
  results_path::String
end

function ROMPath(FEMPaths, RB_method)

  ROM_path = joinpath(FEMPaths.current_test, RB_method)
  create_dir(ROM_path)
  ROM_structures_path = joinpath(ROM_path, "ROM_structures")
  create_dir(ROM_structures_path)
  results_path = joinpath(ROM_path, "results")
  create_dir(results_path)

  ROMPath(ROM_structures_path, results_path)

end

struct ROMInfoS <: ROMInfo
  FEMInfo::FOMInfoS
  Paths::ROMPath
  RB_method::String
  nₛ::Int
  ϵₛ::Float
  use_norm_X::Bool
  online_RHS::Bool
  nₛ_MDEIM::Int
  post_process::Bool
  get_snapshots::Bool
  get_offline_structures::Bool
  save_offline::Bool
  save_online::Bool
end

mutable struct ROMInfoST <: ROMInfo
  FEMInfo::FOMInfoST
  Paths::ROMPath
  RB_method::String
  nₛ::Int
  ϵₛ::Float
  use_norm_X::Bool
  online_RHS::Bool
  nₛ_MDEIM::Int
  nₛ_MDEIM_time::Int
  post_process::Bool
  get_snapshots::Bool
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

function Base.getproperty(RBInfo::ROMInfoS, sym::Symbol)
  if sym in (:affine_structures, :unknowns, :FEM_structures)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end

function Base.getproperty(RBInfo::ROMInfoST, sym::Symbol)
  if sym in (:affine_structures, :unknowns, :FEM_structures)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end


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
