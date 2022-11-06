function get_FEM_snap_path(RBInfo::ROMInfo)
  RBInfo.FEMInfo.Paths.FEM_snap_path
end

function get_FEM_structures_path(RBInfo::ROMInfo)
  RBInfo.FEMInfo.Paths.FEM_structures_path
end

function get_FEM_D(RBInfo::ROMInfo)
  RBInfo.FEMInfo.D
end

function get_μ(RBInfo::ROMInfo)
  get_μ(RBInfo.FEMInfo)
end

function get_FEMμ_info(RBInfo::ROMInfo, ::Val{D}) where D
  get_FEMμ_info(RBInfo.FEMInfo, Val(D))
end

function get_FEMμ_info(RBInfo::ROMInfo, μ::Vector{T}, ::Val{D}) where {D,T}
  get_FEMμ_info(RBInfo.FEMInfo, μ, Val(D))
end

function isaffine(RBInfo::ROMInfo, var::String)
  isaffine(RBInfo.FEMInfo, var)
end

function isaffine(RBInfo::ROMInfo, vars::Vector{String})
  isaffine(RBInfo.FEMInfo, vars)
end

function isnonlinear(::ROMInfo{ID}, var::String) where ID
  if ID == 3
    var ∈ ("C", "D", "LC")
  else
    false
  end
end

function isnonlinear(RBInfo::ROMInfo{ID}, vars::Vector{String}) where ID
  Broadcasting(var->isnonlinear(RBInfo, var))(vars)
end

function islinear(RBInfo::ROMInfo{ID}, args...) where ID
  .!isnonlinear(RBInfo, args...)
end

function get_FEM_vectors(RBInfo::ROMInfo)
  get_FEM_vectors(RBInfo.FEMInfo)::Vector{String}
end

function isvector(RBInfo::ROMInfo, var::String)
  isvector(RBInfo.FEMInfo, var)::Bool
end

function isvector(RBInfo::ROMInfo, vars::Vector{String})
  Broadcasting(var->isvector(RBInfo.FEMInfo, var))(vars)::Vector{Bool}
end

function get_FEM_matrices(RBInfo::ROMInfo)
  get_FEM_matrices(RBInfo.FEMInfo)::Vector{String}
end

function ismatrix(RBInfo::ROMInfo, var::String)
  ismatrix(RBInfo.FEMInfo, var)::Bool
end

function ismatrix(RBInfo::ROMInfo, vars::Vector{String})
  Broadcasting(var->ismatrix(RBInfo.FEMInfo, var))(vars)::Vector{Bool}
end

function get_affine_vectors(RBInfo::ROMInfo)
  get_affine_vectors(RBInfo.FEMInfo)::Vector{String}
end

function get_nonaffine_vectors(RBInfo::ROMInfo)
  get_nonaffine_vectors(RBInfo.FEMInfo)::Vector{String}
end

function get_nonlinear_vectors(RBInfo::ROMInfo)
  idx = findall(x -> isnonlinear(RBInfo, x) .!= 0., get_FEM_vectors(RBInfo))
  get_FEM_vectors(RBInfo)[idx]::Vector{String}
end

function get_linear_vectors(RBInfo::ROMInfo)
  lv = setdiff(get_FEM_vectors(RBInfo), get_nonlinear_vectors(RBInfo))::Vector{String}
  RBInfo.online_RHS ? String[] : lv
end

function get_affine_matrices(RBInfo::ROMInfo)
  get_affine_matrices(RBInfo.FEMInfo)::Vector{String}
end

function get_nonaffine_matrices(RBInfo::ROMInfo)
  get_nonaffine_matrices(RBInfo.FEMInfo)::Vector{String}
end

function get_nonlinear_matrices(RBInfo::ROMInfo)
  idx = findall(x -> isnonlinear(RBInfo, x) .!= 0., get_FEM_matrices(RBInfo))
  get_FEM_matrices(RBInfo)[idx]::Vector{String}
end

function get_linear_matrices(RBInfo::ROMInfo)
  setdiff(get_FEM_matrices(RBInfo), get_nonlinear_matrices(RBInfo))::Vector{String}
end

function get_timesθ(RBInfo::ROMInfo)
  get_timesθ(RBInfo.FEMInfo)::Vector{Float}
end

function assemble_FEM_matrix(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_matrix(FEMSpace, RBInfo.FEMInfo, args...)

end

function assemble_FEM_nonlinear_matrix(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_nonlinear_matrix(FEMSpace, RBInfo.FEMInfo, args...)

end

function assemble_FEM_vector(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_vector(FEMSpace, RBInfo.FEMInfo, args...)

end

function assemble_FEM_nonlinear_vector(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_nonlinear_vector(FEMSpace, RBInfo.FEMInfo, args...)

end

function ParamInfo(
  RBInfo::ROMInfo,
  args...)

  ParamInfo(RBInfo.FEMInfo, args...)

end

function ParamFormInfo(
  RBInfo::ROMInfo,
  args...)

  ParamFormInfo(RBInfo.FEMInfo, args...)

end
