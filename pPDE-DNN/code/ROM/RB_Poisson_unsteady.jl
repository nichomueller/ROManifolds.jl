include("RB_Poisson_steady.jl")


function get_snapshot_matrix(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

    var = "uₕ"
    try
        Sᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * var * ".csv", DataFrame))[:, 1:(ROM_info.nₛ * ROM_info.Nₜ)]
    catch e
        println("Error: $e. Impossible to load the snapshots matrix")
    end
    
    RB_variables.Sᵘ = Sᵘ
    RB_variables.Nᵤˢ = size(Sᵘ)[1]

    @info "Dimension of snapshot matrix: $(size(Sᵘ)); (Nᵤˢ, Nₜ, nₛ) = ($RB_variables.Nᵤˢ, $OM_info.Nₜ, $ROM_info.nₛ)"

end


function PODs_time(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Performing the temporal POD for field u, using a tolerance of $ROM_info.ϵᵗ"

    
    for i in 1:ROM_info.nₜ
        Sᵘₜ = hcat(Sᵘₜ, )

    if ROM_info.time_reduction_technique === "ST-HOSVD"
        Sᵘₜ = zeros(ROM_info.Nₜ, ROM_info.nₜ)

    get_norm_matrix(ROM_info, RB_variables)
    Φₛᵘ = POD(Sᵘ, ROM_info.ϵᵗ)
    nₛᵘ = size(Φₛᵘ)[2]

    return (Φₛᵘ, nₛᵘ)
    
end