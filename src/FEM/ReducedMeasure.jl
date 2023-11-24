function Arrays.lazy_map(k::Reindex{<:Table},::Type{T},j_to_i::AbstractArray) where T
  i_to_v = k.values
  Table(i_to_v[j_to_i])
end

function Arrays.lazy_map(k::Reindex{<:CompressedArray},::Type{T},j_to_i::AbstractArray) where T
  i_to_v = k.values
  values = i_to_v.values
  ptrs = i_to_v.ptrs[j_to_i]
  CompressedArray(values,ptrs)
end

function Arrays.lazy_map(k::Reindex{<:PTArray},j_to_i::AbstractArray)
  map(value -> lazy_map(Reindex(value),j_to_i),k.values)
end

struct RBIntegrationMap <: Map
  cell_to_parent_cell::Vector{Int}
end

for T in (:AbstractArray,:PTArray)
  @eval begin
    function Arrays.lazy_map(
      k::RBIntegrationMap,
      b::$T,
      w::AbstractArray,
      j::AbstractArray)

      cell_to_parent_cell = k.cell_to_parent_cell
      bidx = lazy_map(Reindex(b),cell_to_parent_cell)
      widx = lazy_map(Reindex(w),cell_to_parent_cell)
      jidx = lazy_map(Reindex(j),cell_to_parent_cell)
      lazy_map(IntegrationMap(),bidx,widx,jidx)
    end
  end
end

struct ReducedMeasure
  meas::Measure
  cell_to_parent_cell::Vector{Int}
end

CellData.get_triangulation(rmeas::ReducedMeasure) = view(rmeas.meas.quad.trian,rmeas.cell_to_parent_cell)
get_parent_triangulation(rmeas::ReducedMeasure) = rmeas.meas.quad.trian

function ReducedMeasure(rmeas::ReducedMeasure,trians::Triangulation...)
  @unpack meas,cell_to_parent_cell = rmeas
  ptrian = get_parent_triangulation(rmeas)
  trian = get_triangulation(rmeas)
  for t in trians
    if (==)(t,ptrian;shallow=true)
      @unpack (cell_quad,cell_point,cell_weight,trian,
        data_domain_style,integration_domain_style) = meas.quad
      new_quad = CellQuadrature(cell_quad,cell_point,cell_weight,t,
        data_domain_style,integration_domain_style)
      new_meas = ReducedMeasure(Measure(new_quad),cell_to_parent_cell)
      return new_meas
    end
  end
  @unreachable
end

function CellData.integrate(f::CellField,quad::CellQuadrature,cell_to_parent_cell::Vector{Int})
  trian_f = get_triangulation(f)
  trian_x = get_triangulation(quad)

  msg = """\n
    Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
    triangulations. Verify that either the two objects are defined in the same triangulation
    or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
    """
  @check is_change_possible(trian_f,trian_x) msg

  k = RBIntegrationMap(cell_to_parent_cell)
  b = change_domain(f,quad.trian,quad.data_domain_style)
  x = get_cell_points(quad)
  bx = b(x)
  if quad.data_domain_style == PhysicalDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    lazy_map(k,bx,quad.cell_weight)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    cell_map = get_cell_map(quad.trian)
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(k,bx,quad.cell_weight,cell_Jtx)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == ReferenceDomain()
    cell_map = Fill(GenericField(identity),length(bx))
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(k,bx,quad.cell_weight,cell_Jtx)
  else
    @notimplemented
  end
end

# function CellData.integrate(f::CellField,quad::CellQuadrature,cell_to_parent_cell::Vector{Int})
#   trian_f = get_triangulation(f)
#   trian_x = get_triangulation(quad)

#   msg = """\n
#     Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
#     triangulations. Verify that either the two objects are defined in the same triangulation
#     or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
#     """
#   @check is_change_possible(trian_f,trian_x) msg

#   b = change_domain(f,quad.trian,quad.data_domain_style)
#   x = get_cell_points(quad)
#   bx = b(x)
#   integral = if quad.data_domain_style == PhysicalDomain() &&
#             quad.integration_domain_style == PhysicalDomain()
#     lazy_map(IntegrationMap(),bx,quad.cell_weight)
#   elseif quad.data_domain_style == ReferenceDomain() &&
#             quad.integration_domain_style == PhysicalDomain()
#     cell_map = get_cell_map(quad.trian)
#     cell_Jt = lazy_map(∇,cell_map)
#     cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
#     lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
#   elseif quad.data_domain_style == ReferenceDomain() &&
#             quad.integration_domain_style == ReferenceDomain()
#     cell_map = Fill(GenericField(identity),length(bx))
#     cell_Jt = lazy_map(∇,cell_map)
#     cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
#     lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
#   else
#     @notimplemented
#   end
#   return lazy_map(Reindex(integral),cell_to_parent_cell)
# end

function Arrays.getindex!(cont,a::PTIntegrand,rmeas::ReducedMeasure)
  @unpack meas,cell_to_parent_cell = rmeas
  ptrian = get_parent_triangulation(rmeas)
  trian = get_triangulation(rmeas)
  itrian = get_triangulation(a.meas)
  if (==)(itrian,ptrian)
    integral = integrate(a.object,meas.quad,cell_to_parent_cell)
    add_contribution!(cont,trian,integral)
    return cont
  end
  @unreachable """\n
    There is no contribution associated with the given mesh in this PTIntegrand object.
  """
end

function CellData.integrate(a::PTIntegrand)
  integrate(a.object,a.meas)
end

function CellData.integrate(a::PTIntegrand,meas::ReducedMeasure...)
  @assert length(meas) == 1
  cont = init_contribution(a)
  for m in meas
    getindex!(cont,a,m)
  end
  cont
end

function Arrays.getindex!(cont,a::CollectionPTIntegrand{N},rmeas::ReducedMeasure) where N
  @unpack meas,cell_to_parent_cell = rmeas
  ptrian = get_parent_triangulation(rmeas)
  trian = get_triangulation(rmeas)
  for i = 1:N
    op,int = a[i]
    imeas = int.meas
    itrian = get_triangulation(imeas)
    if (==)(itrian,ptrian)
      integral = integrate(int.object,meas.quad,cell_to_parent_cell)
      add_contribution!(cont,trian,integral,op)
    end
  end
  if num_domains(cont) > 0
    return cont
  end
  @unreachable """\n
    There is no contribution associated with the given mesh in this PTIntegrand object.
  """
end

function CellData.integrate(a::CollectionPTIntegrand,meas::ReducedMeasure...)
  cont = init_contribution(a)
  for m in meas
    getindex!(cont,a,m)
  end
  cont
end
