################################# OFFLINE ######################################

function assemble_constraint_matrix(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}



end

function assemble_supremizers(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}


end

function supr_enrichment(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}


end

function get_RB_space(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}



end

function set_operators(rbinfo::ROMInfo{ID}) where ID



end

function assemble_affine_structure(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Var::VVariable{T}) where {ID,T}


end

function assemble_affine_structure(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Var::MVariable{T}) where {ID,T}


end

function assemble_affine_structure(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}


end

function assemble_MDEIM_structure(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Var::MVVariable{T}) where {ID,T}


end

function assemble_MDEIM_structure(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

end

function assemble_MDEIM_Matₙ(
  Vars::MVariable{T},
  args...) where T


end

function assemble_MDEIM_Matₙ(
  Vars::VVariable{T},
  args...) where T

end

function assemble_offline_structures(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  operators::Vector{String}) where {ID,T}

end

function save_Var_structures(
  rbinfo::ROMInfo{ID},
  Var::MVVariable{T},
  operators::Vector{String}) where {ID,T}

end

function save_offline(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  operators::Vector{String}) where {ID,T}

end

function get_offline_Var(
  rbinfo::ROMInfo{ID},
  Var::MVariable) where ID

end

function get_offline_Var(
  rbinfo::ROMInfo{ID},
  Var::VVariable) where ID


end

function get_offline_Var(
  rbinfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

end

function load_offline(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}


end

################################## ONLINE ######################################

function get_system_blocks(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int}) where {ID,T}

end

function save_system_blocks(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  operators::Vector{String},
  args...) where {ID,T}


end

function assemble_θ(
  FEMSpace::FOM{D},
  rbinfo::ROMInfo{ID},
  Var::MVVariable{T},
  μ::Vector{T}) where {ID,D,T}


end

function assemble_θ(
  FEMSpace::FOM{D},
  rbinfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}},
  μ::Vector{T}) where {ID,D,T}

end

function assemble_θ(
  FEMSpace::FOM{D},
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  μ::Vector{T}) where {ID,D,T}

end

function assemble_θ_function(
  FEMSpace::FOM{D},
  rbinfo::ROMInfo{ID},
  Var::MVVariable{T},
  μ::Vector{T}) where {ID,D,T}

end

function assemble_θ_function(
  FEMSpace::FOM{D},
  rbinfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}},
  μ::Vector{T}) where {ID,D,T}

end

function assemble_θ_function(
  FEMSpace::FOM{D},
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  μ::Vector{T}) where {ID,D,T}


end

function assemble_solve_reconstruct(
  rbinfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  μ::Vector{Vector{T}}) where {ID,T}


end

function save_online(
  rbinfo::ROMInfo{ID},
  offline_time::Float,
  mean_pointwise_err::Matrix{T},
  mean_err::T,
  mean_online_time::Float) where {ID,T}

  times = times_dictionary(rbinfo, offline_time, mean_online_time)
  writedlm(joinpath(rbinfo.results_path, "times.csv"), times)

  path_err = joinpath(rbinfo.results_path, "mean_err.csv")
  save_CSV([mean_err], path_err)

  path_pwise_err = joinpath(rbinfo.results_path, "mean_point_err.csv")
  save_CSV(mean_pointwise_err, path_pwise_err)

  return
end
