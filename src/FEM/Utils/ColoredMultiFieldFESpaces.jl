struct ColoredMultiFieldStyle{N,C} <: MultiFieldStyle end

function ColoredMultiFieldStyle(N::Int,C::NTuple{M,Tuple}) where M
  has_color = fill(false,N)
  for Ci in C
    for c in Ci
      has_color[c] = true
    end
  end
  @check all(has_color)
  return ColoredMultiFieldStyle{N,C}()
end

function ColoredMultiFieldStyle(C::NTuple{M,Tuple}) where M
  N = maximum(map(maximum,C))
  return ColoredMultiFieldStyle(N,C)
end

function ColoredMultiFieldStyle(spaces::Vector{<:ColoredFESpace})
  C = []
  o = 0
  for space in spaces
    n = num_colors(space)
    push!(C,Tuple((1+o:n+o)))
    o += n
  end
  return ColoredMultiFieldStyle(Tuple(C))
end

num_colors(::Type{<:ColoredMultiFieldStyle{N}}) where N = N
num_colors(mfs::ColoredMultiFieldStyle) = num_colors(typeof(mfs))
num_colors(f::MultiFieldFESpace) = num_colors(MultiFieldStyle(f))

get_color_map(::Type{<:ColoredMultiFieldStyle{N,C}}) where {N,C} = C
get_color_map(mfs::ColoredMultiFieldStyle) = get_color_map(typeof(mfs))
get_color_map(f::MultiFieldFESpace) = get_color_map(MultiFieldStyle(f))

function MultiColorFESpace(space::MultiFieldFESpace,reffe::Vector,args...;kwargs...)
  spaces = map(space,reffe) do space, reffe
    MultiColorFESpace(space,reffe,args...;kwargs...)
  end
  VT = get_vector_type(first(spaces))
  N  = num_colors(first(spaces))
  mfs = ColoredMultiFieldStyle(N,Tuple(fill(Tuple(1:N),length(spaces))))
  return MultiFieldFESpace(VT,spaces,mfs)
end

function MultiField.MultiFieldFESpace(
  spaces::Vector{<:ColoredFESpace};
  style = ColoredMultiFieldStyle(spaces)
)
  Ts = map(get_dof_value_type,spaces)
  T  = typeof(*(map(zero,Ts)...))
  VT = BlockVector{T,Vector{Vector{T}}}
  return MultiFieldFESpace(VT,spaces,style)
end

function MultiField.compute_field_offsets(f::MultiFieldFESpace,::ColoredMultiFieldStyle)
  nfields = num_fields(f)
  ncolors = num_colors(f)
  cmap = get_color_map(f)

  offsets = zeros(Int,(nfields,ncolors))
  for i in 1:(nfields-1)
    Ui = f[i]
    dofs_per_color = num_dofs_per_color(Ui)
    offsets[i+1,:] .= offsets[i,:]
    for (j,c) in enumerate(cmap[i])
      offsets[i+1,c] += dofs_per_color[j]
    end
  end

  offsets = map(x -> offsets[x[1],[x[2]...]],enumerate(cmap))
  return offsets
end

function MultiField._restrict_to_field(
  f,
  mfs::ColoredMultiFieldStyle{N,C},
  free_values::BlockVector,
  field
) where {N,C}
  @check blocklength(free_values) == N
  block_ids = [C[field]...]
  return mortar(blocks(free_values)[block_ids])
end

function get_cell_dof_offsets(W::ColoredFESpace,offsets)
  get_cell_dof_offsets(W.space,offsets,get_triangulation(W))
end

function get_cell_dof_offsets(W::ColoredFESpace,offsets::AbstractVector{T},trian::Triangulation) where T <: Integer
  cell_colors = get_cell_dof_color(W,trian)
  k = PosNegReindex(offsets,[zero(T)])
  return lazy_map(Broadcasting(k),cell_colors)
end

function FESpaces.get_free_dof_ids(f::MultiFieldFESpace,::ColoredMultiFieldStyle)
  nfields = num_fields(f)
  ncolors = num_colors(f)
  cmap    = get_color_map(f)
  ndofs_per_field_and_color = map(num_dofs_per_color,f)
  ndofs_per_color = zeros(Int64,ncolors)
  for i in 1:nfields
    for (j,c) in enumerate(cmap[i])
      ndofs_per_color[c] += ndofs_per_field_and_color[i][j]
    end
  end
  return BlockArrays.blockedrange(ndofs_per_color)
end

function FESpaces.get_cell_dof_ids(f::MultiFieldFESpace,
                                   trian::Triangulation,
                                   ::ColoredMultiFieldStyle)
  offsets = MultiField.compute_field_offsets(f)
  nfields = length(f.spaces)
  cmaps = get_color_map(f)
  blockmask = [ is_change_possible(get_triangulation(Vi),trian) for Vi in f.spaces ]
  active_block_ids = findall(blockmask)
  active_block_data = Any[]
  for i in active_block_ids
    cell_dofs_i = recolor(get_cell_dof_ids(f.spaces[i],trian),cmaps[i])
    if i == 1
      push!(active_block_data,cell_dofs_i)
    else
      offset = Int32.(offsets[i])
      cell_ofsets = get_cell_dof_offsets(f.spaces[i],offset,trian)
      cell_dofs_i_b = lazy_map(Broadcasting(MultiField._sum_if_first_positive),cell_dofs_i,cell_ofsets)
      push!(active_block_data,cell_dofs_i_b)
    end
  end
  lazy_map(BlockMap(nfields,active_block_ids),active_block_data...)
end

function Arrays.lazy_map(k::Broadcasting, a::LazyArray{<:Fill{ColorMap}}, offsets::AbstractArray)
  cell_ids = lazy_map(k,a.args[1],offsets)
  cell_colors = a.args[2]
  return lazy_map(ColorMap(),cell_ids,cell_colors)
end

function recolor(a::LazyArray{<:Fill{ColorMap}},colormap::Tuple)
  N = length(colormap)
  if colormap == Tuple(1:N)
    return a
  end
  cell_ids = a.args[1]
  cell_colors = lazy_map(Broadcasting(PosNegReindex(Int8[colormap...],[Int8(-1)])),a.args[2])
  return lazy_map(ColorMap(),cell_ids,cell_colors)
end

# Assembly

function FESpaces.SparseMatrixAssembler(
  mat,vec,
  trial::MultiFieldFESpace{<:ColoredMultiFieldStyle},
  test ::MultiFieldFESpace{<:ColoredMultiFieldStyle},
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

for T in (:AddEntriesMap,:TouchEntriesMap)
  @eval begin
    function Fields.return_cache(k::$T,A::MatrixBlock,v::MatrixBlock,I::VectorBlock{<:ColoredArray},J::VectorBlock{<:ColoredArray})
      qs = findall(v.touched)
      i, j = Tuple(first(qs))
      cij = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
      ni,nj = size(v.touched)
      cache = Matrix{typeof(cij)}(undef,ni,nj)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            cache[i,j] = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
          end
        end
      end
      cache
    end

    function Fields.evaluate!(cache,k::$T,A::MatrixBlock,v::MatrixBlock,I::VectorBlock{<:ColoredArray},J::VectorBlock{<:ColoredArray})
      ni,nj = size(v.touched)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            evaluate!(cache[i,j],k,A,v.array[i,j],I.array[i],J.array[j])
          end
        end
      end
    end

    function Fields.return_cache(k::$T,A::VectorBlock,v::VectorBlock,I::VectorBlock{<:ColoredArray})
      qs = findall(v.touched)
      i = first(qs)
      ci = return_cache(k,A,v.array[i],I.array[i])
      ni = length(v.touched)
      cache = Vector{typeof(ci)}(undef,ni)
      for i in 1:ni
        if v.touched[i]
          cache[i] = return_cache(k,A,v.array[i],I.array[i])
        end
      end
      cache
    end

    function Fields.evaluate!(cache, k::$T,A::VectorBlock,v::VectorBlock,I::VectorBlock{<:ColoredArray})
      ni = length(v.touched)
      for i in 1:ni
        if v.touched[i]
          evaluate!(cache[i],k,A,v.array[i],I.array[i])
        end
      end
    end
  end # @eval
end

# Inject/Prolongate

function prolongate!(x::Vector,Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},y::Vector;dof_ids=LinearIndices(y))
  @error "x should be a BlockVector!"
end
function inject!(x::Vector,Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},y::Vector)
  @error "y should be a BlockVector!"
end
function inject!(x::Vector,Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},y::Vector,w::Vector,w_sums::Vector)
  @error "y should be a BlockVector!"
end

# x \in  MultiColorFESpace
# y \in  original FESpace
function prolongate!(
  x::BlockVector{T},
  Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},
  y::Vector{T};
  dof_ids=LinearIndices(y)
) where T <: Number
  nfields = num_fields(Vh)
  field_to_colors = get_color_map(Vh)
  field_to_dof_to_cdof  = map(get_dof_to_colored_dof,Vh)
  field_to_dof_to_color = map(get_dof_color,Vh)
  field_to_dof_offset   = MultiField._compute_field_offsets(map(Vhi -> Vhi.space,Vh))
  field_to_cdof_offset  = MultiField.compute_field_offsets(Vh)
  x_blocks = blocks(x)
  field = 1
  for dof in dof_ids
    if (field != nfields) && (dof > field_to_dof_offset[field+1])
      field += 1
    end
    dof_offset = field_to_dof_offset[field]
    cdof = field_to_dof_to_cdof[field][dof-dof_offset]
    if cdof > 0
      color = field_to_dof_to_color[field][dof-dof_offset]
      block = field_to_colors[field][color]
      cdof_offset = field_to_cdof_offset[field][color]
      x_blocks[block][cdof+cdof_offset] = y[dof]
    end
  end
end

# x \in  MultiColorFESpace
# y \in  original FESpace
function prolongate!(
  x::BlockVector{T},
  Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},
  y::Vector{T},
  w::BlockVector,
  w_sums::Vector;
  dof_ids=LinearIndices(y)
) where T <: Number
  nfields = num_fields(Vh)
  field_to_colors = get_color_map(Vh)
  field_to_dof_to_cdof  = map(get_dof_to_colored_dof,Vh)
  field_to_dof_to_color = map(get_dof_color,Vh)
  field_to_dof_offset   = MultiField._compute_field_offsets(map(Vhi -> Vhi.space,Vh))
  field_to_cdof_offset  = MultiField.compute_field_offsets(Vh)
  x_blocks = blocks(x)
  w_blocks = blocks(w)
  field = 1
  for dof in dof_ids
    if (field != nfields) && (dof > field_to_dof_offset[field+1])
      field += 1
    end
    dof_offset = field_to_dof_offset[field]
    cdof = field_to_dof_to_cdof[field][dof-dof_offset]
    if cdof > 0
      color = field_to_dof_to_color[field][dof-dof_offset]
      block = field_to_colors[field][color]
      cdof_offset = field_to_cdof_offset[field][color]
      x_blocks[block][cdof+cdof_offset] = y[dof] * w_blocks[block][cdof+cdof_offset] / w_sums[dof]
    end
  end
end

# x \in  original FESpace
# y \in  MultiColorFESpace
function inject!(
  x::Vector{T},
  Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},
  y::BlockVector{T}
) where T <: Number
  nfields = num_fields(Vh)
  field_to_colors = get_color_map(Vh)
  field_to_dof_to_cdof  = map(get_dof_to_colored_dof,Vh)
  field_to_dof_to_color = map(get_dof_color,Vh)
  field_to_ndofs        = map(Vhi -> num_free_dofs(Vhi.space),Vh)
  field_to_dof_offset   = MultiField._compute_field_offsets(map(Vhi -> Vhi.space,Vh))
  field_to_cdof_offset  = MultiField.compute_field_offsets(Vh)
  y_blocks = blocks(y)
  zz = zero(T)
  for field in 1:nfields
    dof_offset = field_to_dof_offset[field]
    for dof in 1:field_to_ndofs[field]
      cdof = field_to_dof_to_cdof[field][dof]
      if cdof > 0
        color = field_to_dof_to_color[field][dof]
        block = field_to_colors[field][color]
        cdof_offset = field_to_cdof_offset[field][color]
        x[dof+dof_offset] = y_blocks[block][cdof+cdof_offset]
      else
        x[dof+dof_offset] = zz
      end
    end
  end
end

function inject!(
  x::Vector{T},
  Vh::MultiFieldFESpace{<:ColoredMultiFieldStyle},
  y::BlockVector{T},
  w::BlockVector,
  w_sums::Vector
) where T <: Number
  nfields = num_fields(Vh)
  field_to_colors = get_color_map(Vh)
  field_to_dof_to_cdof  = map(get_dof_to_colored_dof,Vh)
  field_to_dof_to_color = map(get_dof_color,Vh)
  field_to_ndofs        = map(Vhi -> num_free_dofs(Vhi.space),Vh)
  field_to_dof_offset   = MultiField._compute_field_offsets(map(Vhi -> Vhi.space,Vh))
  field_to_cdof_offset  = MultiField.compute_field_offsets(Vh)
  y_blocks = blocks(y)
  w_blocks = blocks(w)
  zz = zero(T)
  for field in 1:nfields
    dof_offset = field_to_dof_offset[field]
    for dof in 1:field_to_ndofs[field]
      cdof = field_to_dof_to_cdof[field][dof]
      if cdof > 0
        color = field_to_dof_to_color[field][dof]
        block = field_to_colors[field][color]
        cdof_offset = field_to_cdof_offset[field][color]
        x[dof+dof_offset] = y_blocks[block][cdof+cdof_offset] * w_blocks[block][cdof+cdof_offset] / w_sums[dof+dof_offset]
      else
        x[dof+dof_offset] = zz
      end
    end
  end
end
