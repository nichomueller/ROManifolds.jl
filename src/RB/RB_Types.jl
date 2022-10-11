abstract type RBProblem{T} end
abstract type RBProblemS{T} <: RBProblem end
abstract type RBProblemST{T} <: RBProblem end

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
  Matₙ::Matrix{T}
  MDEIM::VMDEIM{T}
end


mutable struct MVariable{T} <: MVVariable{T}
  var::String
  Matₙ::Matrix{T}
  MDEIM::MMDEIM{T}
end

function VVariable(var::String, ::Type{T}) where T

  Matₙ = Matrix{T}(undef,0,0)

  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  MDEIM_Vec = VMDEIM(Mat, Matᵢ, idx, time_idx, el)

  VVariable(var, Matₙ, MDEIM_Vec)

end

function MVariable(var::String, ::Type{T}) where T

  Matₙ = Matrix{T}(undef,0,0)

  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  row_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  MDEIM_Mat = MMDEIM(Mat, Matᵢ, idx, time_idx, row_idx, el)

  MVariable(var, Matₙ, MDEIM_Mat)

end

function MVVariable(Vars::Vector{MVVariable}, var::String)

  for Var in Vars
    if Var.var == var
      return Var
    end
  end

  error("Unrecognized variable")

end

function init_RBVars(::NTuple{1,Int}, ::Type{T}) where T

  S = Matrix{T}[]
  Φₛ = Matrix{T}[]
  x̃ = Matrix{T}[]
  xₙ = Matrix{T}[]
  X₀ = Matrix{SparseMatrixCSC{Float, Int}}[]
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]

  VarA = MVariable("A", T)
  VarF = VVariable("F", T)
  VarH = VVariable("H", T)
  VarL = VVariable("L", T)
  Vars = [VarA, VarF, VarH, VarL]

  Nₛ = Int[]
  nₛ = Int[]
  offline_time = 0.0
  online_time = 0.0

  S, Φₛ, x̃, xₙ, X₀, LHSₙ, RHSₙ, Vars, Nₛ, nₛ, offline_time, online_time

end

function init_RBVars(::NTuple{2,Int}, ::Type{T}) where T

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

mutable struct PoissonS{T} <: RBProblemS{T}
  S::Vector{Matrix{T}}; Φₛ::Vector{Matrix{T}}; x̃::Vector{Matrix{T}}; xₙ::Vector{Matrix{T}};
  X₀::Vector{SparseMatrixCSC{Float64, Int64}}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}};
  Vars::Vector{MVariable{T}}; Nₛ::Vector{Int}; nₛ::Vector{Int};
  offline_time::Float; online_time::Float
end

mutable struct PoissonST{T} <: RBProblemST{T}
  Steady::PoissonS{T}; Φₜᵘ::Vector{Matrix{T}}; Mₙ::Vector{Matrix{T}}; MDEIM_M::MMDEIM{T};
  Nₜ::Int; N::Vector{Int}; nₜ::Vector{Int}; n::Vector{Int}
end

mutable struct StokesS{T} <: RBProblemS{T}
  Poisson::PoissonS{T}; Bₙ::Vector{Matrix{T}}; Lcₙ::Vector{Matrix{T}};
  MDEIM_B::MMDEIM{T}; MDEIM_Lc::VMDEIM{T};
end

mutable struct StokesST{T} <: RBProblemST{T}
  Poisson::PoissonST{T}; Steady::StokesS{T}
end

mutable struct NavierStokesS{T} <: RBProblemS{T}
  Stokes::StokesS{T}; Cₙ::Vector{Matrix{T}}; Dₙ::Vector{Matrix{T}};
  MDEIM_C::MMDEIM{T}; MDEIM_D::MMDEIM{T}
end

mutable struct NavierStokesST{T} <: RBProblemST{T}
  Stokes::StokesST{T}; Steady::NavierStokesS{T};
end

function setup(NT::NTuple{1,Int}, ::Type{T}) where T

  PoissonS{T}(init_RBVars(NT, T)...)

end

function setup(NT::NTuple{2,Int}, ::Type{T}) where T

  PoissonST{T}(
    setup(get_NTuple(1, Int), T), init_RBVars(NT, T)...)

end

function setup(NT::NTuple{3,Int}, ::Type{T}) where T

  StokesS{T}(
    setup(get_NTuple(1, Int), T), init_RBVars(NT, T)...)

end

function setup(NT::NTuple{4,Int}, ::Type{T}) where T

  StokesST{T}(
    setup(get_NTuple(2, Int), T), setup(get_NTuple(3, Int), T))

end

function setup(NT::NTuple{5,Int}, ::Type{T}) where T

  NavierStokesS{T}(
    setup(get_NTuple(3, Int), T), init_RBVars(NT, T)...)

end

function setup(::NTuple{6,Int}, ::Type{T}) where T

  NavierStokesST{T}(
    setup(get_NTuple(4, Int), T), setup(get_NTuple(5, Int), T))

end

function Base.getproperty(RBVars::PoissonST, sym::Symbol)
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Steady, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::PoissonST, sym::Symbol, x::T) where T
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesS, sym::Symbol)
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesS, sym::Symbol, x::T) where T
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
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
  elseif sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
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
  elseif sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Poisson.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesS, sym::Symbol)
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    getfield(RBVars.Stokes.Poisson, sym)
  elseif sym in (:Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc)
    getfield(RBVars.Stokes, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::NavierStokesS, sym::Symbol, x::T) where T
  if sym in (:S, :Φᵘ, :x̃, :xₙ, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛ, :nₛ, :offline_time, :online_time)
    setfield!(RBVars.Stokes.Poisson, sym, x)::T
  elseif sym in (:Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesST, sym::Symbol)
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛ, :p̃, :pₙ, :p̂, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
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
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛ, :p̃, :pₙ, :p̂, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Mₙ, :MDEIM_mat_M, :MDEIMᵢ_M, :MDEIM_idx_M, :row_idx_M, :sparse_el_M,
    :MDEIM_idx_time_A, :MDEIM_idx_time_B, :MDEIM_idx_time_M, :DEIM_idx_time_F, :DEIM_idx_time_H,
    :Nₜ, :Nᵘ, :nₜᵘ, :nᵘ, :Qᵐ, :Φₜᵖ, :Nᵖ, :nₜᵖ, :nᵖ)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

struct ROMPath <: Info
  FEMPaths::FEMPathInfo
  ROM_structures_path::String
  results_path::String
end

struct ROMInfoS{T} <: InfoS
  FEMInfo::FEMInfoS
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

mutable struct ROMInfoST{T} <: InfoST
  FEMInfo::FEMInfoST
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
  if sym in (:probl_nl, :problem_unknowns, :problem_structures)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end

function Base.getproperty(RBInfo::ROMInfoST, sym::Symbol)
  if sym in (:probl_nl, :problem_unknowns, :problem_structures)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end
