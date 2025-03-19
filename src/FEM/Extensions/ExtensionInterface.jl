abstract type ExtensionStyle end
struct ZeroExtension <: ExtensionStyle end
struct FunctionExtension <: ExtensionStyle end
struct HarmonicExtension <: ExtensionStyle end

abstract type Extension{E<:ExtensionStyle} end

struct GenericExtension{E<:ExtensionStyle} <: Extension{E}
  style::E
  matdata
  vecdata
  values::FEFunction
  fdof_to_bg_fdofs::AbstractVector
  ddof_to_bg_ddofs::AbstractVector
end

function Extension(
  style::ZeroExtension,
  space::SingleFieldFESpace,
  fdof_to_bg_fdofs::AbstractVector,
  ddof_to_bg_ddofs::AbstractVector
  )

  zh = zero(space)
  matdata = _mass_data(space)
  vecdata = _interp_data(space,zh)
  GenericExtension(style,matdata,vecdata,zh,fdof_to_bg_fdofs,ddof_to_bg_ddofs)
end

function Extension(
  style::FunctionExtension,
  space::SingleFieldFESpace,
  fdof_to_bg_fdofs::AbstractVector,
  ddof_to_bg_ddofs::AbstractVector,
  f::Function
  )

  fh = interpolate_everywhere(f,space)
  matdata = _mass_data(space)
  vecdata = _interp_data(space,fh)
  GenericExtension(style,matdata,vecdata,fh,fdof_to_bg_fdofs,ddof_to_bg_ddofs)
end

function Extension(
  style::HarmonicExtension,
  space::SingleFieldFESpace,
  fdof_to_bg_fdofs::AbstractVector,
  ddof_to_bg_ddofs::AbstractVector,
  a::Function,
  l::Function,
  )

  matdata = _get_matdata(space,a)
  vecdata = _get_vecdata(space,l)
  assem = SparseMatrixAssembler(space,space)
  A = assemble_matrix(assem,matdata)
  b = assemble_vector(assem,vecdata)
  u = zero_free_values(space)
  solve!(u,LUSolver(),A,b)
  uh = FEFunction(space,u)
  GenericExtension(style,matdata,vecdata,uh,fdof_to_bg_fdofs,ddof_to_bg_ddofs)
end

function Extension(style::ExtensionStyle,bg_space::SingleFieldFESpace,space::SingleFieldFESpace,args...)
  fdof_to_bg_fdofs,ddof_to_bg_ddofs = get_dof_to_bg_dof(bg_space,space)
  Extension(style,space,fdof_to_bg_fdofs,ddof_to_bg_ddofs,args...)
end

function Extension(style::ExtensionStyle,args...)
  @abstractmethod
end

FESpaces.get_fe_space(ext::GenericExtension) = ext.values.fe_space

function FESpaces.get_cell_dof_values(ext::GenericExtension)
  get_cell_dof_values(ext.values)
end

function FESpaces.gather_free_and_dirichlet_values(ext::GenericExtension)
  fv = get_free_dof_values(ext)
  dv = get_dirichlet_dof_values(ext)
  (fv,dv)
end

get_out_fdof_to_bg_fdofs(ext::GenericExtension) = ext.fdof_to_bg_fdofs
get_out_ddof_to_bg_ddofs(ext::GenericExtension) = ext.ddof_to_bg_ddofs

function _mass_data(space::SingleFieldFESpace)
  Ω = get_triangulation(space)
  degree = 2*get_polynomial_order(space)
  dΩ = Measure(Ω,degree)
  a(u,v) = ∫(u⋅v)dΩ
  _get_matdata(space,a)
end

function _interp_data(space::SingleFieldFESpace,uh::FEFunction)
  Ω = get_triangulation(space)
  degree = 2*get_polynomial_order(space)
  dΩ = Measure(Ω,degree)
  l(v) = ∫(uh⋅v)dΩ
  _get_vecdata(space,l)
end

function _get_matdata(space::SingleFieldFESpace,a::Function)
  du = get_trial_fe_basis(space)
  v = get_fe_basis(space)
  matdata = collect_cell_matrix(space,space,a(du,v))
  return matdata
end

function _get_vecdata(space::SingleFieldFESpace,l::Function)
  v = get_fe_basis(space)
  vecdata = collect_cell_vector(space,l(v))
  return vecdata
end
