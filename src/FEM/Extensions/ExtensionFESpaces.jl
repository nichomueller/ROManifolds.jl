struct ExtensionFESpace{S<:SingleFieldFESpace,E<:Extension} <: SingleFieldFESpace
  space::S
  extension::E
  bg_space::SingleFieldFESpace
  dof_to_bg_dofs::AbstractVector
end

function ExtensionFESpace(
  space::SingleFieldFESpace,
  extension::Extension,
  bg_space::SingleFieldFESpace)

  dof_to_bg_dofs = get_dof_to_bg_dof(bg_space,space)
  ExtensionFESpace(space,extension,bg_space,dof_to_bg_dofs)
end

function ZeroExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace
  )

  ext = Extension(ZeroExtension(),bg_space,ext_space)
  ExtensionFESpace(int_space,ext,bg_space,ext_space)
end

function FunctionExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  g::Function)

  ext = Extension(FunctionExtension(),bg_space,ext_space,g)
  ExtensionFESpace(int_space,ext,bg_space,ext_space)
end

function HarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function)

  ext = Extension(HarmonicExtension(),bg_space,ext_space,a,l)
  ExtensionFESpace(int_space,ext,bg_space)
end

FESpaces.ConstraintStyle(::Type{<:ExtensionFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::ExtensionFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::ExtensionFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::ExtensionFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::ExtensionFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::ExtensionFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::ExtensionFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::ExtensionFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::ExtensionFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::ExtensionFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::ExtensionFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::ExtensionFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::ExtensionFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::ExtensionFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::ExtensionFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::ExtensionFESpace) = get_vector_type(f.space)

function FESpaces._cell_vals(f::ExtensionFESpace,object)
  FESpaces._cell_vals(f.bg_space,object)
end

function CellData.CellField(f::ExtensionFESpace,cellvals)
  CellField(f.bg_space,cellvals)
end

function FESpaces.scatter_free_and_dirichlet_values(f::ExtensionFESpace,fdof_to_val,ddof_to_val)
  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  out_bg_cells = get_out_cells_to_bg_cells(f)
  out_out_cells = get_out_cells_to_outcut_cells(f)
  bg_cells = lazy_append(incut_bg_cells,out_bg_cells)

  incut_cell_vals = scatter_free_and_dirichlet_values(f.space,fdof_to_val,ddof_to_val)
  outcut_cell_vals = get_cell_dof_values(f.extension.values)
  out_cell_vals = lazy_map(Reindex(outcut_cell_vals),out_out_cells)
  bg_cell_vals = lazy_append(incut_cell_vals,out_cell_vals)

  lazy_map(Reindex(bg_cell_vals),sortperm(bg_cells))
end

function FESpaces.gather_free_and_dirichlet_values(f::ExtensionFESpace,bg_cell_vals)
  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  cell_vals = lazy_map(Reindex(bg_cell_vals),incut_bg_cells)
  FESpaces.gather_free_and_dirichlet_values(f.space,cell_vals)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fdof_to_val,
  ddof_to_val,
  f::ExtensionFESpace,
  bg_cell_vals)

  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  cell_vals = lazy_map(Reindex(bg_cell_vals),incut_bg_cells)
  gather_free_and_dirichlet_values!(fdof_to_val,ddof_to_val,f.space,cell_vals)
end

function gather_extended_free_and_dirichlet_values(f::ExtensionFESpace,cell_vals)
  bg_fdof_to_val = zero_free_values(f.bg_space)
  bg_ddof_to_val = zero_dirichlet_values(f.bg_space)
  gather_extended_free_and_dirichlet_values!(bg_fdof_to_val,bg_ddof_to_val,f,cell_vals)
end

function gather_extended_free_and_dirichlet_values!(
  bg_fdof_to_val,
  bg_ddof_to_val,
  f::ExtensionFESpace,
  cell_vals)

  gather_free_and_dirichlet_values!(bg_fdof_to_val,bg_ddof_to_val,f.bg_space,cell_vals)
end

# param

function ParamZeroExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace)

  ext = Extension(ParamExtension(ZeroExtension()),bg_space,ext_space,a,l)
  ExtensionFESpace(int_space,ext,bg_space)
end

function ParamFunctionExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  f::Function)

  ext = Extension(ParamExtension(FunctionExtension()),bg_space,ext_space,f)
  ExtensionFESpace(int_space,ext,bg_space)
end

function ParamHarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function)

  ext = Extension(ParamExtension(HarmonicExtension()),bg_space,ext_space,a,l)
  ExtensionFESpace(int_space,ext,bg_space)
end

function Arrays.evaluate(f::ExtensionFESpace{S,UnEvalExtension{E}},args...) where {S,E}
  extension = f.extension(args...)
  ExtensionFESpace(f.space,extension,f.bg_space,f.dof_to_bg_dofs)
end

(space::ExtensionFESpace)(t) = evaluate(space,t)
(space::ExtensionFESpace)(μ,t) = evaluate(space,t)

function ODEs.allocate_space(U::UnEvalTrialFESpace{<:ExtensionFESpace},μ::Realization)
  HomogeneousTrialParamFESpace(U.space(μ),length(μ))
end

function ODEs.allocate_space(U::UnEvalTrialFESpace{<:ExtensionFESpace},μ::Realization,t)
  HomogeneousTrialParamFESpace(U.space(μ,t),length(μ)*length(t))
end

# utils

get_incut_fe_space(f::ExtensionFESpace) = f.space
get_outcut_fe_space(f::ExtensionFESpace) = get_fe_space(f.extension)

function get_incut_cells_to_bg_cells(f::ExtensionFESpace)
  get_cell_to_bg_cell(get_incut_fe_space(f))
end

function get_outcut_cells_to_bg_cells(f::ExtensionFESpace)
  get_cell_to_bg_cell(get_outcut_fe_space(f))
end

function get_cut_cells_to_bg_cells(f::ExtensionFESpace)
  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  outcut_bg_cells = get_outcut_cells_to_bg_cells(f)
  intersect(incut_bg_cells,outcut_bg_cells)
end

function get_in_cells_to_bg_cells(f::ExtensionFESpace)
  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  cut_bg_cells = get_cut_cells_to_bg_cells(f)
  setdiff(incut_bg_cells,cut_bg_cells)
end

function get_out_cells_to_bg_cells(f::ExtensionFESpace)
  outcut_bg_cells = get_outcut_cells_to_bg_cells(f)
  cut_bg_cells = get_cut_cells_to_bg_cells(f)
  setdiff(outcut_bg_cells,cut_bg_cells)
end

function get_bg_cells_to_incut_cells(f::ExtensionFESpace)
  get_bg_cell_to_cell(get_incut_fe_space(f))
end

function get_in_cells_to_incut_cells(f::ExtensionFESpace)
  in_bg_cells = get_in_cells_to_bg_cells(f)
  bg_incut_cells = get_bg_cells_to_incut_cells(f)
  collect(lazy_map(Reindex(bg_incut_cells),in_bg_cells))
end

function get_bg_cells_to_outcut_cells(f::ExtensionFESpace)
  get_bg_cell_to_cell(get_outcut_fe_space(f))
end

function get_out_cells_to_outcut_cells(f::ExtensionFESpace)
  out_bg_cells = get_out_cells_to_bg_cells(f)
  bg_outcut_cells = get_bg_cells_to_outcut_cells(f)
  collect(lazy_map(Reindex(bg_outcut_cells),out_bg_cells))
end

get_out_dof_to_bg_dofs(f::ExtensionFESpace) = get_out_dof_to_bg_dofs(f.extension)
get_in_dof_to_bg_dofs(f::ExtensionFESpace) = f.dof_to_bg_dofs
