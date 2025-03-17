struct SingleFieldExtensionFESpace{CS<:ConstraintStyle,E,V} <: SingleFieldFESpace
  extension::E
  vector_type::Type{V}
  int_space::SingleFieldFESpace
  ext_space::SingleFieldFESpace
  int_dofs_to_bg_dofs::AbstractVector
  ext_dofs_to_bg_dofs::AbstractVector

  function SingleFieldExtensionFESpace(
    extension::E,
    vector_type::Type{V},
    int_space::SingleFieldFESpace,
    ext_space::SingleFieldFESpace,
    int_dofs_to_bg_dofs::AbstractVector,
    ext_dofs_to_bg_dofs::AbstractVector
    ) where {E,V}

    @check isempty(intersect(int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs))
    @check length(int_dofs_to_bg_dofs) == num_free_dofs(int_space)
    @check length(ext_dofs_to_bg_dofs) == num_free_dofs(ext_space)
    CS = typeof(ConstraintStyle(int_space))
    new{CS,E,V}(extension,vector_type,int_space,ext_space,int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs)
  end
end

function SingleFieldExtensionFESpace(
  extension::Extension,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  int_dofs_to_bg_dofs::AbstractVector,
  ext_dofs_to_bg_dofs::AbstractVector)

  zfiv = zero_free_values(int_space)
  zfev = zero_free_values(ext_space)
  zfv = mortar([zfiv,zfev])
  vector_type = typeof(zfv)
  SingleFieldExtensionFESpace(extension,vector_type,int_space,ext_space,
    int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs)
end

function SingleFieldExtensionFESpace(
  extension::Extension,
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace)

  int_dofs_to_bg_dofs = get_dof_to_bg_dof(bg_space,int_space)
  ext_dofs_to_bg_dofs = get_dof_to_bg_dof(bg_space,ext_space)
  SingleFieldExtensionFESpace(extension,int_space,ext_space,int_dofs_to_bg_dofs,ext_dofs_to_bg_dofs)
end

function ZeroExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace
  )

  ext = ZeroExtension(num_free_dofs(ext_space))
  SingleFieldExtensionFESpace(ext,bg_space,int_space,ext_space)
end

function FunctionExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  g::Function)

  ext = FunctionExtension(g,ext_space)
  SingleFieldExtensionFESpace(ext,bg_space,int_space,ext_space)
end

function HarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  A::AbstractMatrix,
  b::AbstractVector)

  ext = HarmonicExtension(A,b)
  SingleFieldExtensionFESpace(ext,bg_space,int_space,ext_space)
end

function HarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function)

  ext = HarmonicExtension(a,l,ext_space)
  SingleFieldExtensionFESpace(ext,bg_space,int_space,ext_space)
end

get_extension(f::SingleFieldExtensionFESpace) = f.extension
get_internal_space(f::SingleFieldExtensionFESpace) = f.int_space
get_external_space(f::SingleFieldExtensionFESpace) = f.ext_space

Base.length(f::SingleFieldExtensionFESpace) = 2

function Base.getindex(f::SingleFieldExtensionFESpace,i)
  if i == 1
    f.int_space
  else i == 2
    f.ext_space
  end
end

function FESpaces.get_triangulation(f::SingleFieldExtensionFESpace)
  @warn "Fetching the triangulation of an SingleFieldExtensionFESpace will likely result in an error"
  int_trian = get_triangulation(f.int_space)
  ext_trian = get_triangulation(f.ext_space)
  lazy_append(int_trian,ext_trian)
end

function FESpaces.num_free_dofs(f::SingleFieldExtensionFESpace)
  num_free_dofs(f.int_space) + num_free_dofs(f.ext_space)
end

function FESpaces.get_free_dof_ids(f::SingleFieldExtensionFESpace)
  int_nf = num_free_dofs(f.int_space)
  ext_nf = num_free_dofs(f.ext_space)
  return BlockArrays.blockedrange([int_nf,ext_nf])
end

function FESpaces.zero_dirichlet_values(f::SingleFieldExtensionFESpace)
  int_zdv = zero_dirichlet_values(f.int_space)
  ext_zdv = zero_dirichlet_values(f.ext_space)
  [int_zdv,ext_zdv]
end

FESpaces.get_dof_value_type(f::SingleFieldExtensionFESpace{CS,E,V}) where {CS,E,V} = eltype(V)

FESpaces.get_vector_type(f::SingleFieldExtensionFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{SingleFieldExtensionFESpace{CS,E,V}}) where {CS,E,V} = CS()

function FESpaces.get_fe_basis(f::SingleFieldExtensionFESpace)
  int_b = get_fe_basis(f.int_space)
  ext_b = get_fe_basis(f.ext_space)
  int_b_comp = MultiField.MultiFieldFEBasisComponent(int_b,1,2)
  ext_b_comp = MultiField.MultiFieldFEBasisComponent(ext_b,2,2)
  MultiField.MultiFieldCellField([int_b_comp,ext_b_comp])
end

function FESpaces.get_trial_fe_basis(f::SingleFieldExtensionFESpace)
  int_b = get_trial_fe_basis(f.int_space)
  ext_b = get_trial_fe_basis(f.ext_space)
  int_b_comp = MultiField.MultiFieldFEBasisComponent(int_b,1,2)
  ext_b_comp = MultiField.MultiFieldFEBasisComponent(ext_b,2,2)
  MultiField.MultiFieldCellField([int_b_comp,ext_b_comp])
end

function FESpaces.FEFunction(f::SingleFieldExtensionFESpace,fv)
  int_ff = FEFunction(f.int_space,fv[Block(1)])
  ext_ff = FEFunction(f.ext_space,fv[Block(2)])
  MultiFieldFEFunction(fv,f,[int_ff,ext_ff])
end

function FESpaces.FEFunction(
  f::SingleFieldExtensionFESpace,
  fv::AbstractVector,
  dv::Vector{<:AbstractVector}
  )

  @check length(dv) == 2
  int_ff = FEFunction(f.int_space,fv[Block(1)],dv[1])
  ext_ff = FEFunction(f.ext_space,fv[Block(2)],dv[2])
  MultiFieldFEFunction(fv,f,[int_ff,ext_ff])
end

function FESpaces.EvaluationFunction(f::SingleFieldExtensionFESpace,fv)
  int_ff = EvaluationFunction(f.int_space,fv[Block(1)])
  ext_ff = EvaluationFunction(f.ext_space,fv[Block(2)])
  MultiFieldFEFunction(fv,f,[int_ff,ext_ff])
end

function CellData.CellField(f::SingleFieldExtensionFESpace,cv)
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
    function $f(f::SingleFieldExtensionFESpace)
      msg = """\n
      This method does not make sense for multi-field
      since each field can be defined on a different triangulation.
      Pass a triangulation in the second argument to get
      the constrain flag for the corresponding cells.
      """
      @notimplemented msg
    end

    function $f(f::SingleFieldExtensionFESpace,trian::Triangulation)
      $f(to_multi_field(f),trian)
    end
  end
end

function FESpaces.interpolate(objects,f::SingleFieldExtensionFESpace)
  interpolate!(objects,zero_free_values(f),f)
end

function FESpaces.interpolate!(objects,fv::AbstractVector,f::SingleFieldExtensionFESpace)
  int_uh = interpolate!(object[1],fv[Block(1)],f.int_space)
  ext_uh = interpolate!(object[2],fv[Block(2)],f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

function FESpaces.interpolate_everywhere(objects,f::SingleFieldExtensionFESpace)
  fv = zero_free_values(f)
  int_uh = interpolate_everywhere!(
    object[1],fv[Block(1)],zero_dirichlet_values(f.int_space),f.int_space)
  ext_uh = interpolate_everywhere!(
    object[2],fv[Block(2)],zero_dirichlet_values(f.ext_space),f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

function FESpaces.interpolate_everywhere!(objects,fv::AbstractVector,dv::Vector,f::SingleFieldExtensionFESpace)
  fv = zero_free_values(f)
  int_uh = interpolate_everywhere!(object[1],fv[Block(1)],dv[1],f.int_space)
  ext_uh = interpolate_everywhere!(object[2],fv[Block(2)],dv[2],f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

function FESpaces.interpolate_dirichlet(objects,f::SingleFieldExtensionFESpace)
  fv = zero_free_values(f)
  int_uh = interpolate_dirichlet!(
    object[1],fv[Block(1)],zero_dirichlet_values(f.int_space),f.int_space)
  ext_uh = interpolate_dirichlet!(
    object[2],fv[Block(2)],zero_dirichlet_values(f.ext_space),f.ext_space)
  MultiFieldFEFunction(fv,f,[int_uh,ext_uh])
end

# utils

function to_multi_field(f::SingleFieldExtensionFESpace{CS,E,V}) where {CS,E,V}
  spaces = [f.int_space,f.ext_space]
  multi_field_style = MultiField.BlockMultiFieldStyle(2)
  MS = typeof(multi_field_style)
  MultiFieldFESpace(V,spaces,multi_field_style)
end

function MultiField.MultiFieldFEFunction(
  fv::AbstractVector,
  f::SingleFieldExtensionFESpace,
  single_fe_functions::Vector{<:SingleFieldFEFunction}
  )

  MultiFieldFEFunction(fv,to_multi_field(f),single_fe_functions)
end
