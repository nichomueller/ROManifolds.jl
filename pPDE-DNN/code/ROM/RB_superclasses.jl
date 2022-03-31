abstract type RB_problem 

end

mutable struct Poisson_RB <: RB_problem
    Sᵘ
    Nᵤˢ
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
    offline_time
end 

function setup_Poisson_RB(problem::Poisson_RB) :: Poisson_RB
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Nᵤˢ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0   
    
    ũ = Float64[]
    uₙ = Float64[]
    û = Float64[]
    
    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    LHSₙ = Matrix{Float64}[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    RHSₙ = Matrix{Float64}[]
    Xᵘ = sparse([],[],[])

    offline_time = 0.0

    return Poisson_RB(Sᵘ, Nᵤˢ, Φₛᵘ, nₛᵘ, ũ, uₙ, û, Aₙ, Aₙ_affine, Aₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, offline_time)

end

mutable struct steady_ADR_RB <: RB_problem
    Sᵘ
    Nᵤˢ
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
    Nᵤˢ = 0
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

    return steady_ADR_RB(Sᵘ, Nᵤˢ, Φₛᵘ, nₛᵘ, ũ, uₙ, û, Aₙ, Aₙ_affine, Aₙ_idx, Bₙ, Bₙ_affine, Bₙ_idx, Cₙ, Cₙ_affine, Cₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, offline_time)

end

mutable struct ADR_RB <: RB_problem
    Sᵘ
    Nᵤˢ
    Φₛᵘ
    nₛᵘ
    Φₜᵘ
    nₜᵘ
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
    Nᵤˢ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0   
    Φₜᵘ = Array{Float64}(undef, 0, 0)
    nₜᵘ = 0
    
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

    return ADR_RB(Sᵘ, Nᵤˢ, Φₛᵘ, nₛᵘ, Φₜᵘ, nₜᵘ, ũ, uₙ, û, Mₙ, Aₙ, Aₙ_affine, Aₙ_idx, Bₙ, Bₙ_affine, Bₙ_idx, Cₙ, Cₙ_affine, Cₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, offline_time)

end

mutable struct steady_Stokes_RB <: RB_problem
    Sᵘ
    Sᵖ
    Nᵤˢ
    Nᵤᵖ
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

mutable struct Stokes_RB <: RB_problem
    Sᵘ
    Sᵖ
    Nᵤˢ
    Nᵤᵖ
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

mutable struct steady_NS_RB <: RB_problem
    Sᵘ
    Sᵖ
    Nᵤˢ
    Nᵤᵖ
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

mutable struct NS_RB <: RB_problem
    Sᵘ
    Sᵖ
    Nᵤˢ
    Nᵤᵖ
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






