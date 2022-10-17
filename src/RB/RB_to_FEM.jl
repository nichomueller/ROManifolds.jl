function get_FEM_snap_path(RBInfo::ROMInfo)
  RBInfo.FEMInfo.Paths.FEM_snap_path
end

function get_FEM_structures_path(RBInfo::ROMInfo)
  RBInfo.FEMInfo.Paths.FEM_structures_path
end

function get_FEM_D(RBInfo::ROMInfo)
  RBInfo.FEMInfo.D
end

function get_FEMμ_info(RBInfo::ROMInfo, ::Val{D}) where D
  get_FEMμ_info(RBInfo.FEMInfo, Val(D))
end

function isaffine(RBInfo::ROMInfo, var::String)
  isaffine(RBInfo.FEMInfo, var)
end

function isaffine(RBInfo::ROMInfo, vars::Vector{String})
  isaffine(RBInfo.FEMInfo, vars)
end

function isnonlinear(::ROMInfo{ID}, var::String) where ID
  if ID == 3
    var ∈ ("C", "D")
  else
    false
  end
end

function isnonlinear(RBInfo::ROMInfo{ID}, vars::Vector{String}) where ID
  Broadcasting(var->isnonlinear(RBInfo, var))(vars)
end

function get_FEM_vectors(RBInfo::ROMInfo)
  get_FEM_vectors(RBInfo.FEMInfo)
end

function isvector(RBInfo::ROMInfo, var::String)
  isvector(RBInfo.FEMInfo, var)
end

function isvector(RBInfo::ROMInfo, vars::Vector{String})
  Broadcasting(var->isvector(RBInfo.FEMInfo, var))(vars)
end

function get_FEM_matrices(RBInfo::ROMInfo)
  get_FEM_matrices(RBInfo.FEMInfo)
end

function ismatrix(RBInfo::ROMInfo, var::String)
  ismatrix(RBInfo.FEMInfo, var)
end

function ismatrix(RBInfo::ROMInfo, vars::Vector{String})
  Broadcasting(var->ismatrix(RBInfo.FEMInfo, var))(vars)
end

function get_affine_vectors(RBInfo::ROMInfo)
  get_affine_vectors(RBInfo.FEMInfo)
end

function get_affine_matrices(RBInfo::ROMInfo)
  get_affine_matrices(RBInfo.FEMInfo)
end

function get_nonaffine_vectors(RBInfo::ROMInfo)
  get_nonaffine_vectors(RBInfo.FEMInfo)
end

function get_nonaffine_matrices(RBInfo::ROMInfo)
  get_nonaffine_matrices(RBInfo.FEMInfo)
end

function get_timesθ(RBInfo::ROMInfo)
  get_timesθ(RBInfo.FEMInfo)
end

function assemble_FEM_matrix(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_matrix(FEMSpace, RBInfo.FEMInfo, args...)

end

function assemble_FEM_vector(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_vector(FEMSpace, RBInfo.FEMInfo, args...)

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
