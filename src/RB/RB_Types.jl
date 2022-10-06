abstract type RBProblem <: Problem end
abstract type RBProblemS{T} <: RBProblem end
abstract type RBProblemST{T} <: RBProblem end
abstract type MDEIMmv{T} end

mutable struct MDEIMv{T} <: MDEIMmv{T}
  Mat::Matrix{T}
  Matᵢ::Matrix{T}
  idx::Vector{Int}
  time_idx::Vector{Int}
  el::Vector{Int}
end


mutable struct MDEIMm{T} <: MDEIMmv{T}
  Mat::Matrix{T}
  Matᵢ::Matrix{T}
  idx::Vector{Int}
  time_idx::Vector{Int}
  row_idx::Vector{Int}
  el::Vector{Int}
end

function init_MDEIMv(::Type{T}) where T
  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  Mat, Matᵢ, idx, time_idx, el
end

function init_MDEIMm(::Type{T}) where T
  Mat = Matrix{T}(undef,0,0)
  Matᵢ = Matrix{T}(undef,0,0)
  idx = Vector{Int}(undef,0)
  time_idx = Vector{Int}(undef,0)
  row_idx = Vector{Int}(undef,0)
  el = Vector{Int}(undef,0)
  Mat, Matᵢ, idx, time_idx, row_idx, el
end

function init_RBVars(::NTuple{1,Int}, ::Type{T}) where T

  Sᵘ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  ũ = Matrix{T}(undef,0,0)
  uₙ = Matrix{T}(undef,0,0)
  û = Matrix{T}(undef,0,0)
  Aₙ = Matrix{T}[]
  Fₙ = Matrix{T}[]
  Hₙ = Matrix{T}[]
  Lₙ = Matrix{T}[]
  X₀ = Matrix{SparseMatrixCSC{Float, Int}}[]
  LHSₙ = Matrix{T}[]
  RHSₙ = Matrix{T}[]
  MDEIM_A = MDEIMm(init_MDEIMm(T)...)
  MDEIM_F = MDEIMv(init_MDEIMv(T)...)
  MDEIM_H = MDEIMv(init_MDEIMv(T)...)
  MDEIM_L = MDEIMv(init_MDEIMv(T)...)
  Nₛᵘ = 0
  nₛᵘ = 0
  Qᵃ = 0
  Qᶠ = 0
  Qʰ = 0
  Qˡ = 0
  offline_time = 0.0
  online_time = 0.0

  (Sᵘ, Φₛᵘ, ũ, uₙ, û, Aₙ, Fₙ, Hₙ, Lₙ, X₀, LHSₙ, RHSₙ, MDEIM_A, MDEIM_F,
  MDEIM_H, MDEIM_L, Nₛᵘ, nₛᵘ, Qᵃ, Qᶠ, Qʰ, Qˡ, offline_time, online_time)

end

function init_RBVars(::NTuple{2,Int}, ::Type{T}) where T

  Φₜᵘ = Matrix{T}(undef,0,0)
  Mₙ = Matrix{T}[]
  MDEIM_M = MDEIMm(init_MDEIMm(T)...)
  Nₜ = 0
  Nᵘ = 0
  nₜᵘ = 0
  nᵘ = 0
  Qᵐ = 0

  Φₜᵘ, Mₙ, MDEIM_M, Nₜ, Nᵘ, nₜᵘ, nᵘ, Qᵐ

end

function init_RBVars(::NTuple{3,Int}, ::Type{T}) where T

  Sᵖ = Matrix{T}(undef,0,0)
  Φₛᵘ = Matrix{T}(undef,0,0)
  p̃ = Matrix{T}(undef,0,0)
  pₙ = Matrix{T}(undef,0,0)
  p̂ = Matrix{T}(undef,0,0)
  Bₙ = Matrix{T}[]
  Lcₙ = Matrix{T}[]
  MDEIM_B = MDEIMm(init_MDEIMm(T)...)
  MDEIM_Lc = MDEIMv(init_MDEIMv(T)...)
  Nₛᵖ = 0
  nₛᵖ = 0
  Qᵇ = 0
  Qˡᶜ = 0

  Sᵖ, Φₛᵘ, p̃, pₙ, p̂, Bₙ, Lcₙ, MDEIM_B, MDEIM_Lc, Nₛᵖ, nₛᵖ, Qᵇ, Qˡᶜ

end

function init_RBVars(::NTuple{4,Int}, ::Type{T}) where T

  Φₜᵖ = Matrix{T}(undef,0,0)
  Nᵖ = 0
  nₜᵖ = 0
  nᵖ = 0

  Φₜᵖ, Nᵖ, nₜᵖ, nᵖ

end

function init_RBVars(::NTuple{5,Int}, ::Type{T}) where T

  Cₙ = Matrix{T}[]
  Dₙ = Matrix{T}[]
  MDEIM_C = MDEIMm(init_MDEIMm(T)...)
  MDEIM_D = MDEIMm(init_MDEIMm(T)...)
  Qᶜ = 0
  Qᵈ = 0

  Cₙ, Dₙ, MDEIM_C, MDEIM_D, Qᶜ, Qᵈ

end

mutable struct PoissonS{T} <: RBProblemS{T}
  Sᵘ::Matrix{T}; Φₛᵘ::Matrix{T}; ũ::Matrix{T}; uₙ::Matrix{T}; û::Matrix{T};
  Aₙ::Vector{Matrix{T}}; Fₙ::Vector{Matrix{T}}; Hₙ::Vector{Matrix{T}}; Lₙ::Vector{Matrix{T}};
  X₀::Vector{SparseMatrixCSC{Float64, Int64}}; LHSₙ::Vector{Matrix{T}}; RHSₙ::Vector{Matrix{T}};
  MDEIM_A::MDEIMm{T}; MDEIM_F::MDEIMv{T}; MDEIM_H::MDEIMv{T}; MDEIM_L::MDEIMv{T};
  Nₛᵘ::Int; nₛᵘ::Int; Qᵃ::Int; Qᶠ::Int; Qʰ::Int; Qˡ::Int;
  offline_time::Float; online_time::Float
end

mutable struct PoissonST{T} <: RBProblemST{T}
  Steady::PoissonS{T}; Φₜᵘ::Matrix{T}; Mₙ::Vector{Matrix{T}}; MDEIM_M::MDEIMm{T};
  Nₜ::Int; Nᵘ::Int; nₜᵘ::Int; nᵘ::Int; Qᵐ::Int;
end

mutable struct StokesS{T} <: RBProblemS{T}
  Poisson::PoissonS{T}; Sᵖ::Matrix{T}; Φₛᵖ::Matrix{T}; p̃::Matrix{T}; pₙ::Matrix{T};
  p̂::Matrix{T}; Bₙ::Vector{Matrix{T}}; Lcₙ::Vector{Matrix{T}};
  MDEIM_B::MDEIMm{T}; MDEIM_Lc::MDEIMv{T}; Nₛᵖ::Int; nₛᵖ::Int; Qᵇ::Int; Qˡᶜ::Int
end

mutable struct StokesST{T} <: RBProblemST{T}
  Poisson::PoissonST{T}; Steady::StokesS{T};
   Φₜᵖ::Matrix{T}; Nᵖ::Int; nₜᵖ::Int; nᵖ::Int
end

mutable struct NavierStokesS{T} <: RBProblemS{T}
  Stokes::StokesS{T}; Cₙ::Vector{Matrix{T}}; Dₙ::Vector{Matrix{T}};
  MDEIM_C::MDEIMm{T}; MDEIM_D::MDEIMm{T}; Qᶜ::Int; Qᵈ::Int;
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
    setup(get_NTuple(2, Int), T), setup(get_NTuple(3, Int), T),
    init_RBVars(NT, T)...)

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
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ,
    :offline_time, :online_time)
    getfield(RBVars.Steady, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::PoissonST, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ,
    :offline_time, :online_time)
    setfield!(RBVars.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesS, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ,
    :offline_time, :online_time)
    getfield(RBVars.Poisson, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesS, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ,
    :offline_time, :online_time)
    setfield!(RBVars.Poisson, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::StokesST, sym::Symbol)
  if sym in (:Sᵖ, :Φₛᵖ, :p̃, :pₙ, :p̂, :Bₙ, :Lcₙ,
    :MDEIM_B, :MDEIM_Lc, :Nₛᵖ, :nₛᵖ, :Qᵇ, :Qˡᶜ)
    getfield(RBVars.Steady, sym)
  elseif sym in (:Φₜᵘ, :Mₙ, :MDEIM_M, :nₜᵘ, :Nₜ, :Nᵘ, :nᵘ, :Qᵐ)
    getfield(RBVars.Poisson, sym)
  elseif sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀,
    :LHSₙ, :RHSₙ, :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L,
    :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :nₛᵘ, :Nₛᵘ, :offline_time, :online_time)
    getfield(RBVars.Poisson.Steady, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::StokesST, sym::Symbol, x::T) where T
  if sym in (:Sᵖ, :Φₛᵖ, :p̃, :pₙ, :p̂, :Bₙ, :Lcₙ,
    :MDEIM_B, :MDEIM_Lc, :Nₛᵖ, :nₛᵖ, :Qᵇ, :Qˡᶜ)
    setfield!(RBVars.Steady, sym, x)::T
  elseif sym in (:Φₜᵘ, :Mₙ, :MDEIM_M, :nₜᵘ, :Nₜ, :Nᵘ, :nᵘ, :Qᵐ)
    setfield!(RBVars.Poisson, sym, x)::T
  elseif sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀,
    :LHSₙ, :RHSₙ, :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L,
    :Qᵃ, :Qᶠ, :Qʰ, :Qˡ, :nₛᵘ, :Nₛᵘ, :offline_time, :online_time)
    setfield!(RBVars.Poisson.Steady, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesS, sym::Symbol)
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ,
    :offline_time, :online_time)
    getfield(RBVars.Stokes.Poisson, sym)
  elseif sym in (:Sᵖ, :Φₛᵖ, :p̃, :pₙ, :p̂, :Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc,
    :Nₛᵖ, :nₛᵖ, :Qᵇ, :Qˡᶜ)
    getfield(RBVars.Stokes, sym)
  else
    getfield(RBVars, sym)
  end
end

function Base.setproperty!(RBVars::NavierStokesS, sym::Symbol, x::T) where T
  if sym in (:Sᵘ, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Fₙ, :Hₙ, :Lₙ, :X₀, :LHSₙ, :RHSₙ,
    :MDEIM_A, :MDEIM_F, :MDEIM_H, :MDEIM_L, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᶠ, :Qʰ, :Qˡ,
    :offline_time, :online_time)
    setfield!(RBVars.Stokes.Poisson, sym, x)::T
  elseif sym in (:Sᵖ, :Φₛᵖ, :p̃, :pₙ, :p̂, :Bₙ, :Lcₙ, :MDEIM_B, :MDEIM_Lc,
    :Nₛᵖ, :nₛᵖ, :Qᵇ, :Qˡᶜ)
    setfield!(RBVars.Stokes, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(RBVars::NavierStokesST, sym::Symbol)
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
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
  if sym in (:Sᵘ, :Sᵘ_quad, :Φₛᵘ, :ũ, :uₙ, :û, :Aₙ, :Bₙ, :Cₙ, :Fₙ, :Hₙ, :Xᵘ₀, :LHSₙ, :RHSₙ,
    :MDEIM_mat_A, :MDEIMᵢ_A, :MDEIM_idx_A, :row_idx_A, :sparse_el_A,
    :MDEIM_mat_B, :MDEIMᵢ_B, :MDEIM_idx_B, :row_idx_B, :sparse_el_B,
    :MDEIM_mat_C, :MDEIMᵢ_C, :MDEIM_idx_C, :row_idx_C, :sparse_el_C,
    :DEIM_mat_F, :DEIMᵢ_F, :DEIM_idx_F, :sparse_el_F, :DEIM_mat_H, :DEIMᵢ_H,
    :DEIM_idx_H, :sparse_el_H, :Nₛᵘ, :nₛᵘ, :Qᵃ, :Qᵇ, :Qᶜ, :Qᶠ, :Qʰ, :offline_time,
    :online_time, :Sᵖ, :Φₛᵘ, :p̃, :pₙ, :p̂, :Xᵘ, :Xᵖ₀, :Nₛᵖ, :nₛᵖ)
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
  save_offline_structures::Bool
  save_results::Bool
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
  save_offline_structures::Bool
  save_results::Bool
  time_reduction_technique::String
  t₀::Float
  tₗ::Float
  δt::Float
  θ::Float
  ϵₜ::Float
  st_M_DEIM::Bool
  functional_M_DEIM::Bool
  adaptivity::Bool
end

function Base.getproperty(RBInfo::ROMInfoS, sym::Symbol)
  if sym in (:probl_nl,)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end

function Base.getproperty(RBInfo::ROMInfoST, sym::Symbol)
  if sym in (:probl_nl,)
    getfield(RBInfo.FEMInfo, sym)
  elseif sym in (:ROM_structures_path, :results_path)
    getfield(RBInfo.Paths, sym)
  else
    getfield(RBInfo, sym)
  end
end
