struct ExtensionFESpace{CS<:ConstraintStyle,V} <: FESpace
  vector_type::Type{V}
  int_space::SingleFieldFESpace
  ext_space::SingleFieldFESpace
  int_dofs_to_bg_dofs::AbstractVector
  ext_dofs_to_bg_dofs::AbstractVector
  ext_values::AbstractVector

  function ExtensionFESpace(
    ::Type{V},
    int_space::SingleFieldFESpace,
    ext_space::SingleFieldFESpace,
    int_dofs_to_bg_dofs::AbstractVector,
    ext_dofs_to_bg_dofs::AbstractVector,
    ext_values::AbstractVector
    ) where V

    @check isempty(intersect(int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs))
    @check length(int_dofs_to_bg_dofs) == num_free_dofs(int_space)
    @check length(ext_dofs_to_bg_dofs) == num_free_dofs(ext_space)
    @check length(ext_values) == num_free_dofs(ext_space)
    CS = ConstraintStyle(int_space)
    new{CS,V}(V,int_space,ext_space,int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs,ext_values)
  end
end

function ExtensionFESpace(
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  int_dofs_to_bg_dofs::AbstractVector,
  ext_dofs_to_bg_dofs::AbstractVector,
  ext_values::AbstractVector)

  zfiv = zero_free_values(int_space)
  zfev = zero_free_values(ext_space)
  zfv = mortar([zfiv,zfev])
  VT = typeof(zfv)
  ExtensionFESpace(VT,int_space,ext_space)
end

function ExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  ext_values::AbstractVector)

  int_dofs_to_bg_dofs = get_dof_to_bg_dof(bg_space,int_space)
  ext_dofs_to_bg_dofs = get_dof_to_bg_dof(bg_space,ext_space)
  ExtensionFESpace(int_space,ext_space,int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs,ext_values)
end

function ZeroExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace)

  z = zero(Float64)
  ext_values = Fill(z,num_free_dofs(ext_space))
  ExtensionFESpace(bg_space,int_space,ext_space,ext_values)
end

function FunctionExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  g::Function)

  ext_gh = interpolate_everywhere(g,ext_space)
  ext_values = get_free_dof_values(ext_gh)
  ExtensionFESpace(bg_space,int_space,ext_space,ext_values)
end

function HarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  ext_laplacian::AbstractMatrix,
  ext_residual::AbstractVector)

  fact = lu(ext_laplacian)
  ext_values = similar(ext_residual)
  ldiv!(ext_values,fact,ext_residual)
  ExtensionFESpace(bg_space,int_space,ext_space,ext_values)
end

function HarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function)

  # we assume no dirichlet boundaries
  ext_laplacian = assemble_matrix(a,ext_space,ext_space)
  ext_residual = assemble_vector(l,ext_space)
  ExtensionFESpace(bg_space,int_space,ext_space,ext_laplacian,ext_residual)
end

Base.length(f::ExtensionFESpace) = 2

function Base.getindex(f::ExtensionFESpace,i)
  if i == 1
    f.int_space
  else i == 2
    f.ext_space
  end
end

function FESpaces.get_triangulation(f::ExtensionFESpace)
  @warn "Fetching the triangulation of an ExtensionFESpace will likely result in an error"
  int_trian = get_triangulation(f.int_space)
  ext_trian = get_triangulation(f.ext_space)
  lazy_append(int_trian,ext_trian)
end

function FESpaces.num_free_dofs(f::ExtensionFESpace)
  num_free_dofs(f.int_space) + num_free_dofs(f.ext_space)
end

function FESpaces.num_free_dofs(f::ExtensionFESpace)
  int_nf = num_free_dofs(f.int_space)
  ext_nf = num_free_dofs(f.ext_space)
  return BlockArrays.blockedrange([int_nf,ext_nf])
end

function FESpaces.zero_dirichlet_values(f::ExtensionFESpace)
  int_zdv = zero_dirichlet_values(f.int_space)
  ext_zdv = zero_dirichlet_values(f.ext_space)
  [int_zdv,ext_zdv]
end

FESpaces.get_dof_value_type(f::ExtensionFESpace{CS,V}) where {CS,V} = eltype(V)

FESpaces.get_vector_type(f::ExtensionFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{ExtensionFESpace{CS,V}}) where {CS,V} = CS()

function FESpaces.get_fe_basis(f::ExtensionFESpace)
  int_b = get_fe_basis(f.int_space)
  ext_b = get_fe_basis(f.ext_space)
  int_b_comp = MultiFieldFEBasisComponent(int_b,1,2)
  ext_b_comp = MultiFieldFEBasisComponent(ext_b,2,2)
  MultiFieldCellField([int_b_comp,ext_b_comp])
end

function FESpaces.get_trial_fe_basis(f::ExtensionFESpace)
  int_b = get_trial_fe_basis(f.int_space)
  ext_b = get_trial_fe_basis(f.ext_space)
  int_b_comp = MultiFieldFEBasisComponent(int_b,1,2)
  ext_b_comp = MultiFieldFEBasisComponent(ext_b,2,2)
  MultiFieldCellField([int_b_comp,ext_b_comp])
end

function FESpaces.FEFunction(f::ExtensionFESpace,fv)
  int_ff = FEFunction(f.int_space,fv[Block(1)])
  ext_ff = FEFunction(f.ext_space,fv[Block(2)])
  MultiFieldFEFunction(fv,f,[int_ff,ext_ff])
end

function FESpaces.FEFunction(
  f::ExtensionFESpace,
  fv::AbstractVector,
  dv::Vector{<:AbstractVector}
  )

  @check length(dv) == 2
  int_ff = FEFunction(f.int_space,fv[Block(1)],dv[1])
  ext_ff = FEFunction(f.ext_space,fv[Block(2)],dv[2])
  MultiFieldFEFunction(fv,f,[int_ff,ext_ff])
end

function FESpaces.EvaluationFunction(f::ExtensionFESpace,fv)
  int_ff = EvaluationFunction(f.int_space,fv[Block(1)])
  ext_ff = EvaluationFunction(f.ext_space,fv[Block(2)])
  MultiFieldFEFunction(fv,f,[int_ff,ext_ff])
end

function CellData.CellField(f::ExtensionFESpace,cv)
  int_cv = lazy_map(a->first(a.array),cv)
  ext_cv = lazy_map(a->last(a.array),cv)
  MultiFieldCellField([int_cv,ext_cv])
end

for f in (
  :(FESpaces.get_cell_isconstrained),
  :(FESpaces.get_cell_is_dirichlet),
  :(FESpaces.get_cell_constraints),
  :(FESpaces.get_cell_dof_ids))
  @eval begin
    function $f(f::ExtensionFESpace)
      msg = """\n
      This method does not make sense for multi-field
      since each field can be defined on a different triangulation.
      Pass a triangulation in the second argument to get
      the constrain flag for the corresponding cells.
      """
      @notimplemented msg
    end

    function $f(f::ExtensionFESpace,trian::Triangulation)
      $f(to_multi_field(f),trian)
    end
  end
end

function FESpaces.interpolate(objects,f::ExtensionFESpace)
  interpolate!(objects,zero_free_values(f),f)
end

function FESpaces.interpolate!(objects,fv::AbstractVector,f::ExtensionFESpace)
  int_uh = interpolate!(object[1],fv[Block(1)],f.int_space)
  ext_uh = interpolate!(object[2],fv[Block(2)],f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

function FESpaces.interpolate_everywhere(objects,f::ExtensionFESpace)
  fv = zero_free_values(f)
  int_uh = interpolate_everywhere!(
    object[1],fv[Block(1)],zero_dirichlet_values(f.int_space),f.int_space)
  ext_uh = interpolate_everywhere!(
    object[2],fv[Block(2)],zero_dirichlet_values(f.ext_space),f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

function FESpaces.interpolate_everywhere!(objects,fv::AbstractVector,dv::Vector,f::ExtensionFESpace)
  fv = zero_free_values(f)
  int_uh = interpolate_everywhere!(object[1],fv[Block(1)],dv[1],f.int_space)
  ext_uh = interpolate_everywhere!(object[2],fv[Block(2)],dv[2],f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

function FESpaces.interpolate_dirichlet(objects,f::ExtensionFESpace)
  fv = zero_free_values(f)
  int_uh = interpolate_dirichlet!(
    object[1],fv[Block(1)],zero_dirichlet_values(f.int_space),f.int_space)
  ext_uh = interpolate_dirichlet!(
    object[2],fv[Block(2)],zero_dirichlet_values(f.ext_space),f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

# utils

function to_multi_field(f::ExtensionFESpace{CS,V}) where {CS,V}
  spaces = [f.int_space,f.ext_space]
  multi_field_style = MultiField.BlockMultiFieldStyle()
  constraint_style = CS()
  MS = typeof(multi_field_style)
  MultiFieldFESpace{MS,CS,V}(V,spaces,multi_field_style,constraint_style)
end

function MultiField.MultiFieldFEFunction(
  fv::AbstractVector,
  f::ExtensionFESpace,
  single_fe_functions::Vector{<:SingleFieldFEFunction}
  )

  mf = to_multi_field(f)
  MultiFieldFEFunction(fv,f,single_fe_functions)
end
