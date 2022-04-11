include("../FEM/FEM_superclasses.jl")
abstract type RBProblem <: Problem end
abstract type RBProblemUnsteady <: RBProblem end
abstract type RBProblem_steady <: RBProblem end
abstract type STGRB <: RBProblem end
abstract type STPGRB <: RBProblem end

Sᵘ = Array{Float64}(undef, 0, 0)
Sᵖ = Array{Float64}(undef, 0, 0)
Nₛᵘ = 0
Nₛᵖ = 0
Φₛᵘ = Array{Float64}(undef, 0, 0)
Φₛᵖ = Array{Float64}(undef, 0, 0)
nₛᵘ = 0
nₛᵖ = 0
Φₜᵘ = Array{Float64}(undef, 0, 0)
Φₜᵖ = Array{Float64}(undef, 0, 0)
nₜᵘ = 0
nₜᵖ = 0

ũ = Float64[]
uₙ = Float64[]
û = Float64[]
p̃ = Float64[]
pₙ = Float64[]
p̂ = Float64[]

Mₙ = Array{Float64}(undef, 0, 0)
Aₙ = Array{Float64}(undef, 0, 0)
Aₙ_affine = Array{Float64}(undef, 0, 0)
Aₙ_idx = Float64[]
Bₙ = Array{Float64}(undef, 0, 0)
Bₙ_affine = Array{Float64}(undef, 0, 0)
Bₙ_idx = Float64[]
Cₙ = Array{Float64}(undef, 0, 0)
Cₙ_affine = Array{Float64}(undef, 0, 0)
Cₙ_idx = Float64[]
LHSₙ = Matrix{Float64}[]
Fₙ = Float64[]
Fₙ_affine = Float64[]
Fₙ_idx = Float64[]
RHSₙ = Matrix{Float64}[]
Xᵘ = sparse([],[],[])
Xᵖ = sparse([],[],[])
Pᵘ_inv = sparse([],[],[])

offline_time = 0.0


mutable struct PoissonSTGRB <: STGRB
    Sᵘ
    Nₛᵘ
    Φₛᵘ
    nₛᵘ
    ũ
    uₙ
    û
    Aₙ
    Aₙ_affine
    Aₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    Pᵘ_inv
    offline_time
end

mutable struct PoissonSTPGRB <: STPGRB
  poissonSTGRB::PoissonSTGRB
  Pᵘ_inv
end

function setup_PoissonSTGRB(empty_struct::PoissonSTGRB) :: PoissonSTGRB
    #=MODIFY
    =#

    return PoissonSTGRB(Sᵘ, Nₛᵘ, Φₛᵘ, nₛᵘ, ũ, uₙ, û, Aₙ, Aₙ_affine, Aₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, Pᵘ_inv, offline_time)

end

function setup_PoissonSTPGRB(empty_struct::PoissonSTPGRB) :: PoissonSTPGRB
  #=MODIFY
  =#

  @forward (PoissonSTPGRB, :poissonSTGRB) PoissonSTGRB

  return PoissonSTPGRB(PoissonSTGRB(Sᵘ, Nₛᵘ, Φₛᵘ, nₛᵘ, ũ, uₙ, û, Aₙ, Aₙ_affine, Aₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, Pᵘ_inv, offline_time), Pᵘ_inv)

end

mutable struct steady_ADR_RB <: RBProblem
    Sᵘ
    Nₛᵘ
    Φₛᵘ
    nₛᵘ
    ũ
    uₙ
    û
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Bₙ
    Bₙ_affine
    Bₙ_idx
    Cₙ
    Cₙ_affine
    Cₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    offline_time
end

function setup_steady_ADR_RB(problem::steady_ADR_RB) :: steady_ADR_RB
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Nₛᵘ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0

    ũ = Float64[]
    uₙ = Float64[]
    û = Float64[]

    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    Bₙ = Array{Float64}(undef, 0, 0)
    Bₙ_affine = Array{Float64}(undef, 0, 0)
    Bₙ_idx = Float64[]
    Cₙ = Array{Float64}(undef, 0, 0)
    Cₙ_affine = Array{Float64}(undef, 0, 0)
    Cₙ_idx = Float64[]
    LHSₙ = Matrix{Float64}[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    RHSₙ = Matrix{Float64}[]
    Xᵘ = sparse([],[],[])

    offline_time = 0.0

    return steady_ADR_RB(Sᵘ, Nₛᵘ, Φₛᵘ, nₛᵘ, ũ, uₙ, û, Aₙ, Aₙ_affine, Aₙ_idx, Bₙ, Bₙ_affine, Bₙ_idx, Cₙ, Cₙ_affine, Cₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, offline_time)

end

mutable struct ADR_RB <: RBProblem
    Sᵘ
    Nₛᵘ
    Φₛᵘ
    nₛᵘ
    Φₜᵘ
    nₜᵘ
    nᵘ
    ũ
    uₙ
    û
    Mₙ
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Bₙ
    Bₙ_affine
    Bₙ_idx
    Cₙ
    Cₙ_affine
    Cₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    offline_time
end

function setup_ADR_RB(problem::steady_ADR_RB) :: steady_ADR_RB
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Nₛᵘ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0
    Φₜᵘ = Array{Float64}(undef, 0, 0)
    nₜᵘ = 0
    nᵘ = 0

    ũ = Float64[]
    uₙ = Float64[]
    û = Float64[]

    Mₙ = Array{Float64}(undef, 0, 0)
    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    Bₙ = Array{Float64}(undef, 0, 0)
    Bₙ_affine = Array{Float64}(undef, 0, 0)
    Bₙ_idx = Float64[]
    Cₙ = Array{Float64}(undef, 0, 0)
    Cₙ_affine = Array{Float64}(undef, 0, 0)
    Cₙ_idx = Float64[]
    LHSₙ = Matrix{Float64}[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    RHSₙ = Matrix{Float64}[]
    Xᵘ = sparse([],[],[])

    offline_time = 0.0

    return ADR_RB(Sᵘ, Nₛᵘ, Φₛᵘ, nₛᵘ, Φₜᵘ, nₜᵘ, nᵘ, ũ, uₙ, û, Mₙ, Aₙ, Aₙ_affine, Aₙ_idx, Bₙ, Bₙ_affine, Bₙ_idx, Cₙ, Cₙ_affine, Cₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, offline_time)

end

mutable struct steady_Stokes_RB <: RBProblem
    Sᵘ
    Sᵖ
    Nₛᵘ
    Nₛᵖ
    Φₛᵘ
    Φₛᵖ
    nₛᵘ
    nₛᵖ
    ũ
    uₙ
    û
    p̃
    pₙ
    p̂
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Bₙ
    Bₙ_affine
    Bₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    Xᵖ
    offline_time
end

function setup_steady_Stokes_RB(problem::steady_Stokes_RB) :: steady_Stokes_RB
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Sᵖ = Array{Float64}(undef, 0, 0)
    Nₛᵘ = 0
    Nₛᵖ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    Φₛᵖ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0
    nₛᵖ = 0

    ũ = Float64[]
    uₙ = Float64[]
    û = Float64[]
    p̃ = Float64[]
    pₙ = Float64[]
    p̂ = Float64[]

    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    Bₙ = Array{Float64}(undef, 0, 0)
    Bₙ_affine = Array{Float64}(undef, 0, 0)
    Bₙ_idx = Float64[]
    LHSₙ = Matrix{Float64}[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    RHSₙ = Matrix{Float64}[]
    Xᵘ = sparse([],[],[])
    Xᵖ = sparse([],[],[])

    offline_time = 0.0

    return steady_Stokes_RB(Sᵘ, Sᵖ, Nₛᵘ, Nₛᵖ, Φₛᵘ, Φₛᵖ, nₛᵘ, nₛᵖ, ũ, uₙ, û, p̃, pₙ, p̂, Aₙ, Aₙ_affine, Aₙ_idx, Bₙ, Bₙ_affine, Bₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, Xᵖ, offline_time)

end

mutable struct Stokes_RB <: RBProblem
    Sᵘ
    Sᵖ
    Nₛᵘ
    Nₛᵖ
    Φₛᵘ
    Φₛᵖ
    nₛᵘ
    nₛᵖ
    Φₜᵘ
    Φₜᵖ
    nₜᵘ
    nₜᵖ
    ũ
    uₙ
    û
    p̃
    pₙ
    p̂
    Mₙ
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Bₙ
    Bₙ_affine
    Bₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    Xᵖ
    offline_time
end

function setup_Stokes_RB(problem::steady_Stokes_RB) :: steady_Stokes_RB
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Sᵖ = Array{Float64}(undef, 0, 0)
    Nₛᵘ = 0
    Nₛᵖ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    Φₛᵖ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0
    nₛᵖ = 0
    Φₜᵘ = Array{Float64}(undef, 0, 0)
    Φₜᵖ = Array{Float64}(undef, 0, 0)
    nₜᵘ = 0
    nₜᵖ = 0

    ũ = Float64[]
    uₙ = Float64[]
    û = Float64[]
    p̃ = Float64[]
    pₙ = Float64[]
    p̂ = Float64[]

    Mₙ = Array{Float64}(undef, 0, 0)
    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    Bₙ = Array{Float64}(undef, 0, 0)
    Bₙ_affine = Array{Float64}(undef, 0, 0)
    Bₙ_idx = Float64[]
    LHSₙ = Matrix{Float64}[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    RHSₙ = Matrix{Float64}[]
    Xᵘ = sparse([],[],[])
    Xᵖ = sparse([],[],[])

    offline_time = 0.0

    return Stokes_RB(Sᵘ, Sᵖ, Nₛᵘ, Nₛᵖ, Φₛᵘ, Φₛᵖ, nₛᵘ, nₛᵖ, Φₜᵘ, Φₜᵖ, nₜᵘ, nₜᵖ, ũ, uₙ, û, p̃, pₙ, p̂, Mₙ, Aₙ, Aₙ_affine, Aₙ_idx, Bₙ, Bₙ_affine, Bₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, Xᵖ, offline_time)
end

mutable struct steady_NS_RB <: RBProblem
    Sᵘ
    Sᵖ
    Nₛᵘ
    Nₛᵖ
    Φₛᵘ
    Φₛᵖ
    nₛᵘ
    nₛᵖ
    ũ
    uₙ
    û
    p̃
    pₙ
    p̂
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Bₙ
    Bₙ_affine
    Bₙ_idx
    Cₙ
    Cₙ_affine
    Cₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    Xᵖ
    offline_time
end

mutable struct NS_RB <: RBProblem
    Sᵘ
    Sᵖ
    Nₛᵘ
    Nₛᵖ
    Φₛᵘ
    Φₛᵖ
    nₛᵘ
    nₛᵖ
    Φₜᵘ
    Φₜᵖ
    nₜᵘ
    nₜᵖ
    ũ
    uₙ
    û
    p̃
    pₙ
    p̂
    Mₙ
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Bₙ
    Bₙ_affine
    Bₙ_idx
    Cₙ
    Cₙ_affine
    Cₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    Xᵖ
    offline_time
end

setup(empty_struct::PoissonSTGRB) = setup_PoissonSTGRB(empty_struct)
setup(empty_struct::PoissonSTPGRB) = setup_PoissonSTPGRB(empty_struct)
setup(problem::steady_ADR_RB) = setup_steady_ADR_RB(problem)
setup(problem::ADR_RB) = setup_ADR_RB(problem)
setup(problem::steady_Stokes_RB) = setup_steady_Stokes_RB(problem)
setup(problem::Stokes_RB) = setup_Stokes_RB(problem)
