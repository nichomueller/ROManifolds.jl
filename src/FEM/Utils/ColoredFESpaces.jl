"""
    struct ColoredFESpace <: SingleFieldFESpace

    ColoredFESpace(model,reffe,colors; kwargs...)

  A `FESpace` whose dofs are split into disjoint sets,each one associated to a domain color.
  Free dofs which do not belong to any color are rejected,and set as homogeneous dirichlet.

  Contains the following fields:
    - `space`: the underlying `FESpace`
    - `colors`: the colors associated to each set of dofs
    - `color_to_ndofs`: the number of dofs associated to each color
    - `dof_to_color`: the color associated to each dof
    - `dof_to_pdof`: the position of each dof within its color group
"""
struct ColoredFESpace{V,A} <: SingleFieldFESpace
  space:: A
  colors:: Vector{Int8}
  cell_to_color::Vector{Int8}
  color_to_ndofs::Vector{Int64}
  dof_to_color::Vector{Int8}
  dof_to_pdof::Vector{Int32}
  function ColoredFESpace(
    space::SingleFieldFESpace,
    colors::Vector{Int8},
    cell_to_color::Vector{Int8},
    color_to_ndofs::Vector{Int64},
    dof_to_color::Vector{Int8},
    dof_to_pdof::Vector{Int32}
  )
    V = typeof(mortar(map(n->zeros(n),color_to_ndofs)))
    A = typeof(space)
    new{V,A}(space,colors,cell_to_color,color_to_ndofs,dof_to_color,dof_to_pdof)
  end
end

function ColoredFESpace(space::FESpace,trian_in::Triangulation)
  trian = get_triangulation(space)
  trian_out = complementary(trian_in,trian)
  colors,color_to_ndofs,dof_to_color = get_dof_to_color(space,trian_in,trian_out)
  cell_to_color = get_cell_to_color(trian,trian_in,trian_out)
  dof_to_pdof = get_dof_to_colored_dof(color_to_ndofs,dof_to_color)
  ColoredFESpace(space,colors,cell_to_color,color_to_ndofs,dof_to_color,dof_to_pdof)
end

# FESpace interface

FESpaces.get_fe_basis(W::ColoredFESpace)       = get_fe_basis(W.space)
FESpaces.get_trial_fe_basis(W::ColoredFESpace) = get_trial_fe_basis(W.space)
FESpaces.ConstraintStyle(W::ColoredFESpace)    = ConstraintStyle(W.space)
CellData.get_triangulation(W::ColoredFESpace)  = get_triangulation(W.space)
FESpaces.get_fe_dof_basis(W::ColoredFESpace)   = get_fe_dof_basis(W.space)
FESpaces.ConstraintStyle(::Type{<:ColoredFESpace{V,A}}) where {V,A} = ConstraintStyle(A)
FESpaces.get_vector_type(::ColoredFESpace{V}) where V = V

function FESpaces.get_free_dof_ids(W::ColoredFESpace)
  return blockedrange(W.color_to_ndofs)
end

function FESpaces.get_cell_dof_ids(W::ColoredFESpace)
  cell_dof_ids  = get_cell_dof_ids(W.space)
  ndir = num_dirichlet_dofs(W.space)
  if ndir > 0
    data = lazy_map(PosNegReindex(W.dof_to_pdof,collect(Int32,-2:-1:-(ndir+1))),cell_dof_ids.data)
  else
    data = lazy_map(Reindex(W.dof_to_pdof),cell_dof_ids.data)
  end
  cell_pdof_ids = Table(data,cell_dof_ids.ptrs)
  cell_dof_to_color = get_cell_dof_color(W)
  return lazy_map(ColorMap(),cell_pdof_ids,cell_dof_to_color)
end

function FESpaces.get_cell_dof_ids(W::ColoredFESpace,trian::Triangulation)
  if !is_active(trian,W)
    color = get_interface_color(W)
    get_color_cell_dof_ids(W,trian,color)
  else
    FESpaces.get_cell_fe_data(get_cell_dof_ids,W,trian)
  end
end

# ColoredFESpace interface

num_colors(W::ColoredFESpace) = length(W.colors)
num_dofs_per_color(W::ColoredFESpace) = W.color_to_ndofs
get_dof_color(W::ColoredFESpace) = W.dof_to_color
get_dof_to_colored_dof(W::ColoredFESpace) = W.dof_to_pdof

get_inside_color(W::ColoredFESpace) = W.colors[1]
get_interface_color(W::ColoredFESpace) = W.colors[2]
get_outside_color(W::ColoredFESpace) = W.colors[3]

"""
    get_cell_dof_color(W::ColoredFESpace)

  Returns the color of each dof in the `ColoredFESpace`,as a cell-wise `Table`.
"""
function get_cell_dof_color(W::ColoredFESpace)
  cell_dof_ids = get_cell_dof_ids(W.space)
  dof_colors = get_dof_color(W)
  ndir = num_dirichlet_dofs(W.space)
  if ndir > 0
    data = lazy_map(PosNegReindex(dof_colors,fill(Int8(-1),ndir)),cell_dof_ids.data)
  else
    data = lazy_map(Reindex(dof_colors),cell_dof_ids.data)
  end
  return Table(data,cell_dof_ids.ptrs)
end

function get_cell_dof_color(W::ColoredFESpace,trian::Triangulation)
  FESpaces.get_cell_fe_data(get_cell_dof_color,W,trian)
end

function get_color_cell_dof_ids(W::ColoredFESpace,color::Integer)
  cell_dof_ids = get_cell_dof_ids(W.space)
  dof_colors = get_dof_color(W)
  dof_to_pdof = lazy_map(ColorMask(color),W.dof_to_pdof,dof_colors)

  ndir = num_dirichlet_dofs(W.space)
  if ndir > 0
    data = lazy_map(PosNegReindex(dof_to_pdof,collect(Int32,-2:-1:-(ndir+1))),cell_dof_ids.data)
  else
    data = lazy_map(Reindex(dof_to_pdof),cell_dof_ids.data)
  end

  cell_pdof_ids = Table(data,cell_dof_ids.ptrs)
  cell_dof_to_color = get_cell_dof_color(W)
  return lazy_map(ColorMap(),cell_pdof_ids,cell_dof_to_color)
end

function get_color_cell_dof_ids(W::ColoredFESpace,trian::Triangulation,color::Integer)
  f = space -> get_color_cell_dof_ids(space,color)
  FESpaces.get_cell_fe_data(f,W,trian)
end

# Assembly

function FESpaces.SparseMatrixAssembler(
  mat,vec,
  trial::ColoredFESpace,
  test ::ColoredFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
)
  matrix_builder = SparseMatrixBuilder(mat)
  vector_builder = ArrayBuilder(vec)

  # Count block rows/cols
  block_rows = blocks(get_free_dof_ids(trial))
  block_cols = blocks(get_free_dof_ids(test))
  @assert length(block_rows) == length(block_cols)
  NB = length(block_rows); SB = Tuple(fill(1,NB)); P  = Tuple(collect(1:NB))

  # Create block assemblers
  block_idx = CartesianIndices((NB,NB))
  block_assemblers = map(block_idx) do idx
    rows = block_rows[idx[1]]
    cols = block_cols[idx[2]]
    FESpaces.GenericSparseMatrixAssembler(matrix_builder,vector_builder,rows,cols,strategy)
  end

  return MultiField.BlockSparseMatrixAssembler{NB,NB,SB,P}(block_assemblers)
end

# This should be moved to Gridap/BlockAssemblers
FESpaces.map_cell_rows(strategy::MatrixBlock{FESpaces.DefaultAssemblyStrategy},cell_ids) = cell_ids
FESpaces.map_cell_cols(strategy::MatrixBlock{FESpaces.DefaultAssemblyStrategy},cell_ids) = cell_ids

# Prolongate/inject

function prolongate!(x::Vector,Vh::ColoredFESpace,y::Vector;dof_ids=LinearIndices(y))
  @error "x should be a BlockVector!"
end
function inject!(x::Vector,Vh::ColoredFESpace,y::Vector)
  @error "y should be a BlockVector!"
end
function inject!(x::Vector,Vh::ColoredFESpace,y::Vector,w::Vector,w_sums::Vector)
  @error "y should be a BlockVector!"
end

# x \in  ColoredFESpace
# y \in  original FESpace
function prolongate!(
  x::BlockVector{T},Vh::ColoredFESpace,y::AbstractVector{T};dof_ids=LinearIndices(y)
) where T <: Number
  dof_to_pdof = Vh.dof_to_pdof
  dof_to_color  = Vh.dof_to_color
  x_blocks = blocks(x)
  for dof in dof_ids
    pdof = dof_to_pdof[dof]
    if pdof > 0
      color = dof_to_color[dof]
      x_blocks[color][pdof] = y[dof]
    end
  end
end

# x \in  ColoredFESpace
# y \in  original FESpace
function prolongate!(
  x::BlockVector{T},Vh::ColoredFESpace,y::Vector{T},w::BlockVector,w_sums::Vector;dof_ids=LinearIndices(y)
) where T <: Number
  dof_to_pdof = Vh.dof_to_pdof
  dof_to_color  = Vh.dof_to_color
  x_blocks = blocks(x)
  w_blocks = blocks(w)
  for dof in dof_ids
    pdof = dof_to_pdof[dof]
    if pdof > 0
      color = dof_to_color[dof]
      x_blocks[color][pdof] = y[dof] * w_blocks[color][pdof] / w_sums[dof]
    end
  end
end

# x \in  original FESpace
# y \in  ColoredFESpace
function inject!(
  x::AbstractVector{T},Vh::ColoredFESpace,y::BlockVector{T}
) where T <: Number
  dof_to_pdof = Vh.dof_to_pdof
  dof_to_color  = Vh.dof_to_color
  y_blocks = blocks(y)
  zz = zero(T)
  for (dof,pdof) in enumerate(dof_to_pdof)
    if pdof > 0
      color = dof_to_color[dof]
      x[dof] = y_blocks[color][pdof]
    else
      x[dof] = zz
    end
  end
end

function inject!(
  x::AbstractVector{T},Vh::ColoredFESpace,y::BlockVector{T},w::BlockVector,w_sums::AbstractVector
) where T <: Number
  dof_to_pdof = Vh.dof_to_pdof
  dof_to_color  = Vh.dof_to_color
  y_blocks = blocks(y)
  w_blocks = blocks(w)
  zz = zero(T)
  for (dof,pdof) in enumerate(dof_to_pdof)
    if pdof > 0
      color = dof_to_color[dof]
      x[dof] = y_blocks[color][pdof] * w_blocks[color][pdof] / w_sums[dof]
    else
      x[dof] = zz
    end
  end
end

# Auxiliary functions

function get_dof_to_colored_dof(color_to_ndofs,dof_to_color)
  dof_to_colored_dof = Vector{Int32}(undef,length(dof_to_color))
  color_counters = fill(0,length(color_to_ndofs))
  for (dof,color) in enumerate(dof_to_color)
    if color > 0
      color_counters[color] += 1
      dof_to_colored_dof[dof] = color_counters[color]
    else
      dof_to_colored_dof[dof] = -1
    end
  end
  @assert color_counters == color_to_ndofs
  return dof_to_colored_dof
end

function get_dof_to_color(space::FESpace,trian_in::Triangulation,trian_out::Triangulation)
  ncolors = 3
  colors = collect(Int8,1:ncolors) # in, interface, out
  ndofs = num_free_dofs(space)

  reject_ndofs = ndofs
  color_to_ndofs = Vector{Int64}(undef,ncolors)
  dof_to_color   = fill(Int8(-1),ndofs)

  dof_to_mask_in = get_dof_mask(space,trian_in)
  dof_to_mask_out = get_dof_mask(space,trian_out)
  dof_to_mask_Γ = dof_to_mask_in .&& dof_to_mask_out
  dof_to_mask_in[findall(dof_to_mask_Γ)] .= false
  dof_to_mask_out[findall(dof_to_mask_Γ)] .= false

  for (i,dof_to_mask) in enumerate((dof_to_mask_in,dof_to_mask_Γ,dof_to_mask_out))
    dof_to_color[dof_to_mask] .= i
    i_ndofs = sum(dof_to_mask)
    color_to_ndofs[i] = i_ndofs
    reject_ndofs   -= i_ndofs
  end
  @assert sum(color_to_ndofs) == (ndofs-reject_ndofs) "There is overlapping between the colors!"

  return colors,color_to_ndofs,dof_to_color
end

function get_dof_mask(space::FESpace,trian::Triangulation{Dc}) where Dc
  tface_to_mface = get_tface_to_mface(trian)

  cell_dof_ids = get_cell_dof_ids(space)

  cell_dof_ids_cache = array_cache(cell_dof_ids)
  dof_to_mask = fill(false,num_free_dofs(space))
  for cell in tface_to_mface
    dofs = getindex!(cell_dof_ids_cache,cell_dof_ids,cell)
    for dof in dofs
      if dof > 0
        dof_to_mask[dof] = true
      end
    end
  end

  return dof_to_mask
end

function get_cell_to_color(trian::Triangulation,trian_in::Triangulation,trian_out::Triangulation)
  cell_to_color = fill(Int8(-1),num_cells(trian))
  cell_to_mask_in = get_cell_mask(trian,trian_in)
  cell_to_mask_out = get_cell_mask(trian,trian_out)
  cell_to_mask_Γ = cell_to_mask_in .&& cell_to_mask_out
  cell_to_mask_in[findall(cell_to_mask_Γ)] .= false
  cell_to_mask_out[findall(cell_to_mask_Γ)] .= false
  for (i,cell_to_mask) in enumerate((cell_to_mask_in,cell_to_mask_Γ,cell_to_mask_out))
    cell_to_color[cell_to_mask] .= i
  end
  return cell_to_color
end

function get_cell_mask(trian::Triangulation,t::Triangulation)
  @check is_included(t,trian)
  cell_to_mask = fill(false,num_cells(trian))
  tface_to_mface = get_tface_to_mface(t)
  cell_to_mask[tface_to_mface] .= true
  return cell_to_mask
end

function complementary(trian_in::Triangulation,trian::Triangulation)
  tface_to_mface = get_tface_to_mface(trian)
  tface_in_to_mface = get_tface_to_mface(trian_in)
  tface_out_to_mface = setdiff(tface_to_mface,tface_in_to_mface)
  view(trian,tface_out_to_mface)
end

function is_active(trian::Triangulation,W::ColoredFESpace)
  color = get_outside_color(W)
  cells_to_color = W.cell_to_color
  cells = get_tface_to_mface(trian)
  !( all(cells_to_color[cells] .== color) )
end
