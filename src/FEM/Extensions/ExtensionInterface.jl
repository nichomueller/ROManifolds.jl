abstract type Extension end

get_extension_values(ext::Extension) = @abstractmethod
get_extension_vector(ext::Extension) = get_extension_values(ext)
get_extension_matrix(ext::Extension) = @abstractmethod

FESpaces.num_rows(ext::Extension) = size(get_extension_matrix(ext),1)
FESpaces.num_cols(ext::Extension) = size(get_extension_matrix(ext),2)

struct ZeroExtension <: Extension
  n::Int
end

get_extension_values(ext::ZeroExtension) = Fill(zero(Float64),ext.n)
get_extension_matrix(ext::ZeroExtension) = spdiagm(ones(Float64,ext.n))

struct FunctionExtension <: Extension
  values::AbstractVector
end

get_extension_values(ext::FunctionExtension) = ext.values
get_extension_matrix(ext::FunctionExtension) = spdiagm(ones(Float64,ext.n))

function FunctionExtension(g::Function,ext_space::SingleFieldFESpace)
  gh = interpolate_everywhere(g,ext_space)
  values = get_free_dof_values(gh)
  FunctionExtension(values)
end

struct HarmonicExtension <: Extension
  laplacian::AbstractMatrix
  vector::AbstractVector
  values::AbstractVector
end

function HarmonicExtension(laplacian::AbstractMatrix,vector::AbstractVector)
  factor = lu(laplacian)
  values = similar(vector)
  ldiv!(values,factor,vector)
  HarmonicExtension(laplacian,vector,values)
end

# we assume no dirichlet boundaries, so that only a test `ext_space` is needed
function HarmonicExtension(
  a::Function,
  l::Function,
  ext_space::SingleFieldFESpace
  )

  laplacian = assemble_matrix(a,ext_space,ext_space)
  vector = assemble_vector(l,ext_space)
  HarmonicExtension(laplacian,vector)
end

get_extension_values(ext::HarmonicExtension) = ext.values
get_extension_matrix(ext::HarmonicExtension) = ext.laplacian
get_extension_vector(ext::HarmonicExtension) = ext.vector
