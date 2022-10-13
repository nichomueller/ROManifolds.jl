function get_FEM_snap_path(RBInfo::ROMInfo)
  RBInfo.FEMInfo.Paths.FEM_snap_path
end

function get_FEM_structures_path(RBInfo::ROMInfo)
  RBInfo.FEMInfo.Paths.FEM_structures_path
end

function get_FEMμ_info(RBInfo::ROMInfo)
  get_FEMμ_info(RBInfo.FEMInfo)
end

function isaffine(RBInfo::ROMInfo, var::String)
  isaffine(RBInfo.FEMInfo, var)
end

function get_FEM_vectors(RBInfo::ROMInfo)
  get_FEM_vectors(RBInfo.FEMInfo)
end

function isvector(RBInfo::ROMInfo, var::String)
  isvector(RBInfo.FEMInfo, var)
end

function get_FEM_matrices(RBInfo::ROMInfo)
  get_FEM_matrices(RBInfo.FEMInfo)
end

function ismatrix(RBInfo::ROMInfo, var::String)
  ismatrix(RBInfo.FEMInfo, var)
end

function get_timesθ(RBInfo::ROMInfo)
  get_timesθ(RBInfo.FEMInfo)
end

function assemble_FEM_structure(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  args...)

  assemble_FEM_structure(FEMSpace, RBInfo.FEMInfo, args...)

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
