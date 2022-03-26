include("../../../UTILS/general.jl")

const problem_name = "POISSON"   # POISSON / STOKES / NAVIER-STOKES / BURGERS
const problem_type = "STEADY"  # STEADY / UNSTEADY
const approx_type = "ROM"   # ROM / DL
const problem_dim = 3   
const problem_nonlinearities = Dict("Ω" => false, "A" => false, "f" => false, "g" => false, "h" => false)  # true / false
const number_coupled_blocks = 1
const mesh_name = "model.json"
const nₛ = 100
const order = 1
const dirichlet_tags = ["sides"]
const neumann_tags = ["circle", "triangle", "square"]
const solver = "lu"

@assert !(problem_name === "BURGERS" && problem_type === "STEADY")

const root = "/home/user1/git_repos/Mabla.jl/pPDE-NN/code"

function FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim)
    nonlins = ""
    for (key, value) in problem_nonlinearities
        if value === true
            nonlins *= "_" * key
        end
    end
    mesh_path = joinpath(root, joinpath("FEM", joinpath("models", mesh_name)))
    root_test = joinpath(root, joinpath("TESTS", joinpath(problem_type, joinpath(problem_name, joinpath(string(problem_dim) * "D" * nonlins)))))
    FEM_path = joinpath(root_test, joinpath(mesh_name, "FEM_data"))
    FEM_snap_path = joinpath(FEM_path, "snapshots")
    FEM_structures_path = joinpath(FEM_path, "FEM_structures")
    FOM_files_extension = "txt" 
    FOM_files_delimiter = ","
    (out) -> (mesh_path; FEM_snap_path; FEM_structures_path; FOM_files_extension; FOM_files_delimiter)
end 
(out) -> (mesh_path; FEM_snap_path; FEM_structures_path; FOM_files_extension; FOM_files_delimiter)

struct problem_specifics
    problem_name::String
    problem_type::String
    paths::Function
    approx_type::String
    problem_dim::Int
    problem_nonlinearities::Dict
    number_coupled_blocks::Int
    order::Int
    dirichlet_tags::Array
    neumann_tags::Array
    solver::String
    #time_marching_scheme::String
end

struct parametric_specifics{F<:Function}
    params::F
    get_domain::F
    get_α::F
    get_f::F
    get_g::F
    get_h::F
end

#= struct parametric_specifics
    parameters::Function
    model::Function
    data::Function
end =#


function get_domain(b::Bool, μᵒ = nothing)
    @assert !(b === true && μᵒ === nothing) "Provide valid parameter value for the domain deformation, or set Ω to be non-parametric"  
    if b === false
        model = DiscreteModelFromFile(problem_info.paths.mesh_path)    
    else
        #model = generate_cartesian_model(reference_info, deformation, μᵒ) # incomplete: implement domain deformation
        model = generate_cartesian_model(μᵒ)
    end
    model 
end

function get_α(b::Bool, μᴬ::Vector)
    if b == false
        α(x) = sum(μᴬ)
    else
        α(x) = (b === true) * μᴬ[3] + 1 / μᴬ[3] * exp(-((x[1] - μᴬ[1])^2 + (x[2] - μᴬ[2])^2) / μᴬ[3]) + (b === false) * sum(μᴬ)
    end
end

function get_f(b::Bool, μᶠ = nothing)
    @assert !(b === true && μᶠ === nothing) "Provide valid parameter value for the forcing term, or set f to be non-parametric" 
    if b == false
        f(x) = 1
    else
        f(x) = [sin(μᶠ .* x[i]) for i in 1:length(x)]
    end
    f
end
#= function get_parametric_f(x::Point, μᶠ::Float64)
    sin(μᶠ .* x)
end
get_f(x::Point) = get_nonparametric_f(x::Point)
get_f(x::Point, μᶠ::Float64) = get_parametric_f(x, μᶠ)=#

function get_g(b::Bool, μᵍ = nothing)
    @assert !(b === true && μᵍ === nothing) "Provide valid parameter value for the forcing term, or set g to be non-parametric" 
    if b == false
        g(x) = 1
    else
        g(x) = [sin(μᵍ .* x[i]) for i in 1:length(x)]
    end
    g
end

function get_h(b::Bool, μʰ = nothing)
    @assert !(b === true && μʰ === nothing) "Provide valid parameter value for the forcing term, or set h to be non-parametric" 
    if b == false
        h(x) = 1
    else
        h(x) = [sin(μʰ .* x[i]) for i in 1:length(x)]
    end
    h
end

#= function get_nonparametric_g(x::Point)
    1
end
function get_parametric_g(x::Point, μᵍ::Float64)
    sin(μᵍ .* x)
end
get_g(x::Point) = get_nonparametric_g(x::Point)
get_g(x::Point, μᵍ::Float64) = get_parametric_g(x, μᵍ) =#

#= function get_nonparametric_h(x::Point)
    1
end
function get_parametric_h(x::Point, μʰ::Float64)
    sin(μʰ .* x)
end
get_h(x::Point) = get_nonparametric_h(x::Point)
get_h(x::Point, μʰ::Float64) = get_parametric_h(x, μʰ)
 =#

#= function get_parametric_data(problem_nonlinearities::Dict, params::Function, i_nₛ::Int)

    α(x) = αμ(x, Point(params.σ[i_nₛ, :]), problem_nonlinearities["A"])
    f(x) = fμ(x, params.μᶠ[i_nₛ], problem_nonlinearities["f"])
    g(x) = gμ(x, params.μᵍ[i_nₛ], problem_nonlinearities["g"])
    h(x) = hμ(x, params.μʰ[i_nₛ], problem_nonlinearities["h"])

    (out) -> (α; f; g; h)
end
(out) -> (α; f; g; h)
 =#

function generate_parameters(problem_nonlinearities::Dict, nₛ::Int, r::Dict)

    if problem_nonlinearities["Ω"] === true
        μᵒ = generate_value(r["μᵒ"][1], r["μᵒ"][2], nₛ, joinpath(all_paths.FEM_snap_path, "params.jld"), "μᵒ")
        if !isfile(joinpath(all_paths.FEM_snap_path, "params.jld"))
            save(joinpath(all_paths.FEM_snap_path, "params.jld"), "μᵒ", μᵒ)
        end
    else
        μᵒ = generate_value()
    end

    μᴬ = hcat(rand(Uniform(r["μᴬ"][1], r["μᴬ"][2]), nₛ), rand(Uniform(r["μᴬ"][3], r["μᴬ"][4]), nₛ), rand(Uniform(r["μᴬ"][5], r["μᴬ"][6]), nₛ)) 
    if !isfile(joinpath(all_paths.FEM_snap_path, "params.jld"))
        save(joinpath(all_paths.FEM_snap_path, "params.jld"), "μᴬ", μᴬ)
    end
   
    if problem_nonlinearities["f"] === true
        μᶠ = generate_value(r["μᶠ"][1], r["μᶠ"][2], nₛ, joinpath(all_paths.FEM_snap_path, "params.jld"), "μᶠ")
        if !isfile(joinpath(all_paths.FEM_snap_path, "params.jld"))
            save(joinpath(all_paths.FEM_snap_path, "params.jld"), "μᶠ", μᶠ)
        end
    else
        μᶠ = generate_value()
    end
    
    if problem_nonlinearities["g"] === true
        μᵍ = generate_value(r["μᵍ"][1], r["μᵍ"][2], nₛ, joinpath(all_paths.FEM_snap_path, "params.jld"), "μᵍ")
        if !isfile(joinpath(all_paths.FEM_snap_path, "params.jld"))
            save(joinpath(all_paths.FEM_snap_path, "params.jld"), "μᵍ", μᵍ)
        end
    else
        μᵍ = generate_value()
    end

    if problem_nonlinearities["h"] === true
        μʰ = generate_value(r["μʰ"][1], r["μʰ"][2], nₛ, joinpath(all_paths.FEM_snap_path, "params.jld"), "μʰ")
        if !isfile(joinpath(all_paths.FEM_snap_path, "params.jld"))
            save(joinpath(all_paths.FEM_snap_path, "params.jld"), "μʰ", μʰ)
        end
    else
        μʰ = generate_value()
    end

    (out) -> (σ; μᵒ; μᶠ; μᵍ; μʰ)
end
(out) -> (σ; μᵒ; μᶠ; μᵍ; μʰ)


function compute_parametric_info(problem_nonlinearities, params, i_nₛ)
    parametric_specifics(params, 
                        get_domain(problem_nonlinearities["Ω"], get_index(params.μᵒ, i_nₛ)), 
                        get_α(problem_nonlinearities["A"], get_index(params.μᴬ, i_nₛ)), 
                        get_f(problem_nonlinearities["f"], get_index(params.μᶠ, i_nₛ)), 
                        get_g(problem_nonlinearities["g"], get_index(params.μᵍ, i_nₛ)), 
                        get_h(problem_nonlinearities["h"], get_index(params.μʰ, i_nₛ)))
end





